# app.py — Chat-First Anki Deck Creator
# Conversational UX (topic decided in chat) + Optional Manual Mode
# Works WITH or WITHOUT uploaded docs. RAG optional (auto-off when no materials).
# Preview → Iterate → Build .apkg • Variety beyond Q&A • Safe tag sanitization • TTS optional.

import os, io, json, time, tempfile, hashlib, re, random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI

# Core deps
import genanki
from gtts import gTTS
from gtts.lang import tts_langs
from pypdf import PdfReader
import docx as docxlib
from unidecode import unidecode

# Optional deps
try:
    import requests
    from bs4 import BeautifulSoup
    try:
        from readability import Document  # readability-lxml
    except Exception:
        Document = None
except Exception:
    requests = None
    BeautifulSoup = None
    Document = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import pandas as pd
except Exception:
    pd = None

# -------------------------
# App config & theming
# -------------------------
st.set_page_config(page_title="🧠 Chat Anki Deck Creator", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.small { color:#666; font-size:.9rem; }
.kpi { border:1px solid #eee; border-radius:12px; padding:10px; text-align:center; }
.kpi h3 { margin:0; font-size:.9rem; color:#555; }
.kpi .val { font-weight:700; font-size:1.1rem; }
.card-prev { border:1px solid #e9e9ef; border-radius:12px; padding:10px; margin-bottom:10px; }
.hint { color:#666; }
hr { border:none; border-top:1px solid #eee; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# API key & client
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Set **OPENAI_API_KEY** in Streamlit Secrets or environment.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

# -------------------------
# Constants
# -------------------------
TEXT_MODEL = "gpt-4o-mini"
MAX_AUDIO_FILES = 24
AUDIO_CHAR_LIMIT = 400

# -------------------------
# System prompts (chat-first planner + generator)
# -------------------------
SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva que atua como **Criador de Baralhos Anki**.
Trabalhe em 3 passos: (1) entender pedido e propor um plano, (2) gerar prévia variada e específica, (3) produzir o baralho final.

Regras para os cartões:
- Recordação ativa e conhecimento atômico (uma ideia por cartão).
- Perguntas específicas, resposta única, linguagem clara, concisa.
- Variedade além de Q&A: cloze (um único alvo), cenários/mini-casos, procedimentos/passos, contrastes/exceções.
- Para políticas/exames: prazos, responsáveis, thresholds, exceções, auditoria, conformidade.
- Para idiomas: vocabulário (collocations), gramática/padrões (contraste), listening/pronúncia (script natural/IPA quando útil).
- Quando houver materiais (payload.materiais_digest), ancore cartões neles e preencha source_ref.file/page_or_time.
- NÃO dependa de materiais: se não houver, gere a partir do tópico e das metas.

Saída estritamente em JSON:
{
 "deck":{"title":"string","language":"string","level":"string","topic":"string","source_summary":"string","card_count_planned":number},
 "cards":[
   {"id":"string","type":"basic|reverse|cloze","front":"string","back":"string","hint":"string|null",
    "examples":[{"text":"string","translation":"string|null","notes":"string|null"}],
    "audio_script":"string|null","tags":["string"],"difficulty":"easy|medium|hard",
    "source_ref":{"file":"string|null","page_or_time":"string|null","span":"string|null"},
    "rationale":"string"}],
 "qa_report":{"duplicates_removed":number,"too_broad_removed":number,"notes":"string"}
}
Defina deck.card_count_planned = len(cards).
"""

ASSESSMENT_SYSTEM = """
Você é um especialista em Design Instrucional. Analise o pedido do usuário e (se houver) um digest dos materiais.
Proponha um plano sucinto para o baralho, em JSON:
{
 "summary":"string",
 "key_issues":["..."],
 "card_type_strategy":{"basic":int,"reverse":int,"cloze":int,"scenario":int,"procedure":int},
 "anchoring_plan":"string",
 "terminology_focus":["..."],
 "open_questions":["..."]
}
Percentuais ~somam 100 (ignore scenario/procedure se não aplicável).
"""

# -------------------------
# Utils
# -------------------------
def slugify(text: str, maxlen: int = 64) -> str:
    t = unidecode(text or "").lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t[:maxlen] or f"slug-{int(time.time())}"

def html_escape(s: Optional[str]) -> str:
    if not s: return ""
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def normalize_text_for_html(s: str, cloze: bool = False) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"^```[\s\S]*?\n|```$", "", s)
    if cloze:
        s = s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    else:
        s = html_escape(s)
    s = s.replace("\n","<br>")
    return s

def strip_html_to_plain(s: str) -> str:
    if not s: return ""
    s2 = re.sub(r"<[^>]+>", " ", s)
    s2 = (s2.replace("&nbsp;"," ").replace("&amp;","&")
              .replace("&lt;","<").replace("&gt;",">"))
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2[:AUDIO_CHAR_LIMIT]

def examples_to_html(examples: Optional[List[Dict[str, Any]]]) -> str:
    if not examples: return ""
    items = []
    for ex in examples:
        if not isinstance(ex, dict): continue
        line = html_escape(ex.get("text",""))
        tr = ex.get("translation"); nt = ex.get("notes")
        if tr: line += f' <span class="tr">({html_escape(tr)})</span>'
        if nt: line += f' <span class="nt">[{html_escape(nt)}]</span>'
        items.append(f"<li>{line}</li>")
    return "<ul class='examples-list'>" + "".join(items) + "</ul>"

# gTTS
def map_lang_for_gtts(language_code: str) -> str:
    langs = tts_langs()
    if not language_code: return "en"
    lc = language_code.lower()
    prefs = {
        "fr": ["fr"], "pt": ["pt","pt-br"], "en": ["en","en-us","en-gb"],
        "es": ["es"], "de": ["de"], "it": ["it"],
        "zh": ["zh","zh-cn","cmn-cn","zh-tw"], "ja": ["ja"],
    }
    for k, choices in prefs.items():
        if lc.startswith(k):
            for code in choices:
                if code in langs: return code
    for cand in (lc, lc.replace("_","-"), lc.split("-")[0]):
        if cand in langs: return cand
    return "en"

def synth_audio(text: str, lang_code: str) -> Optional[bytes]:
    t = (text or "").strip()
    if not t: return None
    try:
        fp = io.BytesIO()
        gTTS(text=t, lang=map_lang_for_gtts(lang_code)).write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None

# Tags
def _clean_tag(tag) -> str:
    t = str(tag or "").strip().lower()
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"[^a-z0-9_\-:]", "", t)
    return t[:40]
def sanitize_tags(tags) -> list:
    if not isinstance(tags, list): return []
    out, seen = [], set()
    for t in tags:
        ct = _clean_tag(t)
        if ct and ct not in seen:
            out.append(ct); seen.add(ct)
        if len(out) >= 12: break
    return out

# Q/A orientation
QUESTION_WORDS = {
    "fr": ["qui","que","quoi","quel","quelle","quels","quelles","où","quand","comment","pourquoi","combien"],
    "pt": ["o que","qual","quais","quando","onde","como","por que","por quê","quem","quanto","quantos","quantas"],
    "en": ["what","which","when","where","how","why","who","whom","whose"],
}
def _lang_prefix(lang: str) -> str:
    l = (lang or "en").lower()
    for k in ["fr","pt","en","es","de","it"]:
        if l.startswith(k): return k
    return "en"
def looks_like_question(text: str, lang: str) -> bool:
    if not text: return False
    t = text.strip()
    if "?" in t: return True
    lp = _lang_prefix(lang)
    w = re.sub(r'^[\-\•\–\s]+', '', t.lower())
    return any(w.startswith(q + " ") or w == q for q in QUESTION_WORDS.get(lp, []))
def orient_q_a(card: dict, lang: str) -> Tuple[str, str]:
    f = (card.get("front") or card.get("Text") or "").strip()
    b = (card.get("back")  or card.get("BackExtra") or "").strip()
    fq, bq = looks_like_question(f, lang), looks_like_question(b, lang)
    if fq and not bq: return f, b
    if bq and not fq: return b, f
    if len(f) > len(b): return b, f
    return f, b

def anchored_to_source(card: dict) -> bool:
    src = card.get("source_ref") or {}
    return bool(src.get("file") or src.get("page_or_time"))

# -------------------------
# Ingest: files & URLs
# -------------------------
def extract_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes); tmp.flush()
        reader = PdfReader(tmp.name)
        out = []
        for i, p in enumerate(reader.pages, start=1):
            t = p.extract_text() or ""
            if t.strip(): out.append(f"[p.{i}]\n{t}")
        return "\n\n".join(out)
def extract_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes); tmp.flush()
        doc = docxlib.Document(tmp.name)
        return "\n".join(p.text for p in doc.paragraphs)
def extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def ingest_files(uploaded_files) -> List[Dict[str, Any]]:
    mats = []
    for f in uploaded_files or []:
        name = f.name; data = f.read(); low = name.lower()
        if low.endswith(".pdf"): text = extract_pdf(data)
        elif low.endswith(".docx"): text = extract_docx(data)
        elif low.endswith((".txt",".md",".markdown")): text = extract_txt(data)
        else: continue
        mats.append({"file": name, "content": text})
    return mats

def fetch_url_text(url: str, timeout: int = 25) -> str:
    if not requests: return ""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"anki-deck-creator/1.0"})
        r.raise_for_status()
        html = r.text
        if Document:
            try:
                doc = Document(html)
                content = doc.summary(html_partial=False)
                soup = BeautifulSoup(content, "html.parser")
                return soup.get_text("\n", strip=True)
            except Exception:
                pass
        if BeautifulSoup:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script","style","noscript"]): tag.decompose()
            return soup.get_text("\n", strip=True)
        return html
    except Exception:
        return ""

def ingest_urls(urls: List[str]) -> List[Dict[str, Any]]:
    mats = []
    for u in urls:
        u = u.strip()
        if not u: continue
        txt = fetch_url_text(u)
        if txt:
            name = re.sub(r"^https?://", "", u)[:80]
            mats.append({"file": name, "content": txt})
    return mats

def split_sections(text: str, file_name: str) -> List[Dict[str, Any]]:
    sections = []
    parts = re.split(r"\n(?=#+\s)|\n(?=\[p\.\d+\])", text)
    for i, part in enumerate(parts):
        pt = part.strip()
        if not pt: continue
        m = re.match(r"^(#+\s.*)", pt)
        if m:
            title = m.group(1)
        elif pt.lower().startswith("[p."):
            title = pt.split("\n",1)[0]
        else:
            title = f"{file_name} — sec {i+1}"
        sections.append({"file": file_name, "title": title, "content": pt[:4000]})
    return sections or [{"file": file_name, "title": f"{file_name} — full", "content": text[:4000]}]

def simple_keyword_rank(query: str, sections: List[Dict[str, Any]], top_k: int = 6) -> List[int]:
    q_toks = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", query.lower()))
    scores = []
    for i, s in enumerate(sections):
        txt = s["content"].lower()
        score = sum(1 for t in q_toks if t in txt)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:top_k]]

def tfidf_rank(query: str, sections: List[Dict[str, Any]], top_k: int = 6) -> List[int]:
    docs = [s["content"] for s in sections]
    try:
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9)
        X = vec.fit_transform(docs)
        qv = vec.transform([query])
        sims = cosine_similarity(qv, X).ravel()
        top = sims.argsort()[::-1][:top_k]
        return top.tolist()
    except Exception:
        return simple_keyword_rank(query, sections, top_k)

def rag_digest(materials: List[Dict[str, Any]], topic: str, user_feedback: str, top_k: int = 6, max_chars: int = 12000) -> Tuple[str, List[Dict[str, Any]]]:
    if not materials: return "", []
    sections = []
    for m in materials:
        sections.extend(split_sections(m["content"], m["file"]))
    query = f"{topic}\n{user_feedback}".strip()
    idxs = tfidf_rank(query, sections, top_k) if SKLEARN_OK else simple_keyword_rank(query, sections, top_k)
    chosen = [sections[i] for i in idxs]
    parts = []
    for s in chosen:
        hdr = f"# {s['file']} — {s['title']}"
        parts.append(hdr + "\n" + s["content"])
    digest = "\n\n".join(parts)
    return digest[:max_chars], chosen

def compress_materials_simple(materials: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    if not materials: return ""
    per = max(800, max_chars // max(1, len(materials)))
    parts = []
    for m in materials:
        chunk = (m["content"] or "")[:per]
        parts.append(f"# {m['file']}\n{chunk}")
    return "\n\n".join(parts)[:max_chars]

# -------------------------
# OpenAI helpers
# -------------------------
def chat_json(messages, model=TEXT_MODEL, temperature=0, max_tries=4):
    last_err = None
    for i in range(max_tries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type":"json_object"},
            )
            return resp
        except Exception as e:
            last_err = e
            time.sleep(min(8, 2**i + random.random()))
    raise last_err

def _safe_json(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        return {}

# -------------------------
# Validation & variety
# -------------------------
REQUIRED_CARD_FIELDS = {"type","front","back"}

def valid_card(c: dict) -> bool:
    if not isinstance(c, dict): return False
    typ = (c.get("type") or "").lower()
    if typ == "cloze":
        c.setdefault("front", c.get("Text",""))
        c.setdefault("back",  c.get("BackExtra",""))
    if not all(k in c for k in REQUIRED_CARD_FIELDS): return False
    if len(strip_html_to_plain(c.get("back",""))) > 420:  # atomicity
        return False
    if typ == "cloze" and re.findall(r"\{\{c\d+::", c.get("front","")).count("{{") > 1:
        return False
    return True

def dedupe_cards(cards: list) -> list:
    def sig(c):
        t = (c.get("type") or "").lower().strip()
        f = (c.get("front") or c.get("Text") or "").strip().lower()
        b = (c.get("back") or c.get("BackExtra") or "").strip().lower()
        return f"{t}|{f[:160]}|{b[:160]}"
    seen, out = set(), []
    for c in cards or []:
        if not isinstance(c, dict): continue
        s = sig(c)
        if s in seen: continue
        seen.add(s); out.append(c)
    return out

def contains_domain_keyword(card: dict, kws: List[str]) -> bool:
    if not kws: return True
    blob = f"{card.get('front','')} || {card.get('back','')}".lower()
    return any(k in blob for k in kws[:40])

def kind_of(c: dict) -> str:
    t = (c.get("type") or "").lower()
    if t == "cloze": return "cloze"
    f = (c.get("front") or "").lower()
    if "?" in f: return "qa"
    if any(k in f for k in ["scenario:", "cenário:", "case:", "caso:"]): return "scenario"
    if any(k in f for k in ["procedure","procedimento","step","passo","ordem"]): return "procedure"
    return "basic"

def enforce_variety(cards: List[dict], kws: List[str], max_qa_frac: float, require_anchor_effective: bool, seed_ids: set) -> List[dict]:
    # Filter by anchoring/keywords (but don't drop user-seeded IDs)
    filtered = []
    for c in cards:
        if require_anchor_effective and not anchored_to_source(c) and c.get("id") not in seed_ids:
            continue
        if not contains_domain_keyword(c, kws) and c.get("id") not in seed_ids:
            continue
        filtered.append(c)
    if not filtered: return []

    total = max(1, len(filtered))
    qa_idx = [i for i,c in enumerate(filtered) if kind_of(c) == "qa" and c.get("id") not in seed_ids]
    max_qa = int(max_qa_frac * total)
    if len(qa_idx) > max_qa:
        drop = set(qa_idx[max_qa:])
        filtered = [c for i,c in enumerate(filtered) if i not in drop]
    return filtered

# -------------------------
# OpenAI calls (deck generation & assessment)
# -------------------------
def gerar_baralho(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = chat_json(
        [{"role":"system","content":SYSTEM_PROMPT},
         {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
        model=TEXT_MODEL, temperature=0
    )
    data = _safe_json(resp.choices[0].message.content or "{}")
    cards = data.get("cards", [])
    if not isinstance(cards, list): cards = []
    cards = [c for c in cards if valid_card(c)]

    for c in cards:
        typ = (c.get("type") or "basic").lower()
        if typ == "cloze":
            if not c.get("front") and c.get("Text"): c["front"] = c["Text"]
            if not c.get("back")  and c.get("BackExtra"): c["back"] = c["BackExtra"]
            c["front"] = normalize_text_for_html(c.get("front",""), cloze=True)
            c["back"]  = normalize_text_for_html(c.get("back",""))
        else:
            c["front"] = normalize_text_for_html(c.get("front",""))
            c["back"]  = normalize_text_for_html(c.get("back",""))

    data.setdefault("deck", {})
    data["deck"].setdefault("title", payload.get("deck_title","Anki Deck"))
    data["deck"].setdefault("language", payload.get("idioma_alvo","en"))
    data["deck"].setdefault("level", payload.get("nivel_proficiencia",""))
    data["deck"].setdefault("topic", payload.get("topico",""))
    data["deck"]["card_count_planned"] = len(cards)
    data["cards"] = cards
    return data

def generate_assessment(topic: str, goal: str, idioma: str, nivel: str, tipos: list, digest: str, domain_kws: List[str], max_qa_frac: float, require_anchor: bool) -> Dict[str, Any]:
    payload = {
        "topic": topic, "goal": goal, "language": idioma, "level": nivel,
        "allowed_types": tipos, "materials_digest_excerpt": (digest or "")[:6000],
        "domain_keywords": domain_kws[:30], "max_qa_pct": max_qa_frac, "require_anchor": require_anchor
    }
    resp = chat_json(
        [{"role":"system","content":ASSESSMENT_SYSTEM},
         {"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
        model=TEXT_MODEL, temperature=0
    )
    data = _safe_json(resp.choices[0].message.content or "{}")
    data.setdefault("summary","")
    data.setdefault("key_issues",[])
    data.setdefault("card_type_strategy",{"basic":35,"reverse":10,"cloze":35,"scenario":10,"procedure":10})
    data.setdefault("anchoring_plan","Reference file + page/section when available. If no materials, rely on topic-only generation.")
    data.setdefault("terminology_focus",domain_kws[:12])
    data.setdefault("open_questions",[])
    return data

# -------------------------
# Models & Deck building
# -------------------------
def stable_model_id(name: str, version: int = 3) -> int:
    h = hashlib.sha1(f"{name}-v{version}".encode("utf-8")).hexdigest()
    return int(h[:10], 16)

COMMON_CSS = """
.card { font-family: -apple-system, Segoe UI, Roboto, Arial; font-size: 20px; text-align: left; color: #222; background: #fff; }
.front { font-size: 1.05em; line-height: 1.45; }
.back  { line-height: 1.5; }
.hdr  { font-weight: 600; opacity: .8; margin: 6px 0 4px; }
.hint { margin-top: 8px; font-size: 0.95em; color: #666; }
.examples { margin-top: 12px; }
.examples-list { padding-left: 18px; }
.tr { color: #444; font-style: italic; }
.nt { color: #666; }
.extra { margin-top: 10px; font-size: 0.9em; color: #444; }
.audio { margin-top: 8px; }
hr { margin: 14px 0; }
"""

MODEL_BASIC = genanki.Model(
    stable_model_id("Anki-Chat Basic", version=3), "Anki-Chat Basic (v3)",
    fields=[{"name":"Front"},{"name":"Back"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Card 1",
        "qfmt":"<div class='hdr'>Q:</div><div class='front'>{{Front}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"<div class='hdr'>A:</div><div class='back'>{{Back}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div class='audio'>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
               "<hr><div class='hdr'>Q:</div><div class='front'>{{Front}}</div>"
    }],
    css=COMMON_CSS
)

MODEL_REVERSE = genanki.Model(
    stable_model_id("Anki-Chat Reverse", version=3), "Anki-Chat Reverse (v3)",
    fields=[{"name":"Front"},{"name":"Back"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Reverse Only",
        "qfmt":"<div class='hdr'>Q:</div><div class='front'>{{Back}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"<div class='hdr'>A:</div><div class='back'>{{Front}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div class='audio'>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
               "<hr><div class='hdr'>Q:</div><div class='front'>{{Back}}</div>"
    }],
    css=COMMON_CSS
)

MODEL_CLOZE = genanki.Model(
    stable_model_id("Anki-Chat Cloze", version=3), "Anki-Chat Cloze (v3)",
    fields=[{"name":"Text"},{"name":"BackExtra"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Cloze Card",
        "qfmt":"<div class='hdr'>Q:</div><div class='front'>{{cloze:Text}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"<div class='hdr'>A:</div><div class='back'>{{BackExtra}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/
