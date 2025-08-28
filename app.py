# app.py — Chat-first Anki Deck Generator
# Conversational UX • Plan proposal & approval • RAG over uploads • Preview & edit • Final .apkg
# TTS (optional) • Variety beyond Q&A • Tags sanitized • Org/policy & exam modes

import os, io, json, time, tempfile, hashlib, re, random, textwrap
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

# Optional deps (auto-fallback if missing)
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
st.set_page_config(page_title="🧠 Anki Deck Assistant", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.small { color:#666; font-size:.9rem; }
.kpi { border:1px solid #eee; border-radius:12px; padding:10px; text-align:center; }
.kpi h3 { margin:0; font-size:.9rem; color:#555; }
.kpi .val { font-weight:700; font-size:1.1rem; }
.chat-note { font-size:.9rem; color:#555; }
.card-prev { border:1px solid #e9e9ef; border-radius:12px; padding:10px; margin-bottom:10px; }
.hl { background:#fff8d6; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# API key & client
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Set OPENAI_API_KEY in Streamlit Secrets or env first.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

# -------------------------
# Constants
# -------------------------
TEXT_MODEL = "gpt-4o-mini"
MAX_AUDIO_FILES = 24
AUDIO_CHAR_LIMIT = 400

# -------------------------
# Prompts
# -------------------------
SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva, integrada ao 'Anki-Generator'.
Fale de forma colaborativa e prática. Objetivo: propor, iterar e gerar baralhos Anki de alta qualidade.

Princípios essenciais para os cartões:
- Recordação ativa e conhecimento atômico (uma ideia por cartão).
- Perguntas específicas, resposta única, linguagem clara, exemplos concretos.
- Variedade além de Q&A: cloze (um alvo), cenários/mini-casos, procedimentos/passos, contrastes e exceções.
- Para políticas/concursos: prazos, responsáveis, thresholds, exceções, auditoria, conformidade.
- Para idiomas: vocabulário (collocations), gramática/padrões (contraste), listening/pronúncia (IPA/roteiro natural).

Quando houver materiais (payload.materiais_digest), ancore cartões neles e preencha source_ref.file/page_or_time.

Formato de saída (JSON):
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
Você é um especialista em Design Instrucional. Analise o pedido e (se houver) um digest dos materiais.
Proponha um plano de design para o baralho.

Responda APENAS em JSON:
{
 "summary":"string",
 "key_issues":["..."],
 "assumptions":["..."],
 "card_type_strategy":{"basic":int,"reverse":int,"cloze":int,"scenario":int,"procedure":int},
 "anchoring_plan":"string",
 "terminology_focus":["..."],
 "open_questions":["..."],
 "risks":["..."],
 "success_criteria":["..."]
}
Percentuais somam ~100 (ignore scenario/procedure se não aplicável).
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
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"anki-deck-assistant/1.0"})
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
        title = None
        m = re.match(r"^(#+\s.*)", pt)
        if m: title = m.group(1)
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
# Generation validation & variety
# -------------------------
REQUIRED_CARD_FIELDS = {"type","front","back"}

def valid_card(c: dict) -> bool:
    if not isinstance(c, dict): return False
    typ = (c.get("type") or "").lower()
    if typ == "cloze":
        c.setdefault("front", c.get("Text",""))
        c.setdefault("back",  c.get("BackExtra",""))
    if not all(k in c for k in REQUIRED_CARD_FIELDS): return False
    # one fact per card heuristic
    if len(strip_html_to_plain(c.get("back",""))) > 420:
        return False
    # single cloze target
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

def goal_mix_minima(goal: str) -> Dict[str, float]:
    g = (goal or "").lower()
    if "policy" in g: return {"cloze": 0.25, "scenario": 0.35, "procedure": 0.25}
    if "exam" in g:   return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}
    if "vocabulary" in g: return {"cloze": 0.35, "scenario": 0.05, "procedure": 0.00}
    if "grammar" in g:    return {"cloze": 0.50, "scenario": 0.10, "procedure": 0.00}
    return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}

def contains_domain_keyword(card: dict, kws: List[str]) -> bool:
    if not kws: return True
    blob = f"{card.get('front','')} || {card.get('back','')}".lower()
    return any(k in blob for k in kws[:40])

def enforce_variety(cards: List[dict], kws: List[str], max_qa_frac: float, require_anchor: bool, seed_ids: set) -> List[dict]:
    def kind(c):
        t = (c.get("type") or "").lower()
        if t == "cloze": return "cloze"
        f = (c.get("front") or "").lower()
        if "?" in f: return "qa"
        if any(k in f for k in ["scenario:", "cenário:", "case:", "caso:"]): return "scenario"
        if any(k in f for k in ["procedure","procedimento","step","passo","ordem"]): return "procedure"
        return "basic"

    filtered = []
    for c in cards:
        if require_anchor and not anchored_to_source(c) and c.get("id") not in seed_ids:
            continue
        if not contains_domain_keyword(c, kws) and c.get("id") not in seed_ids:
            continue
        filtered.append(c)
    if not filtered: return []

    total = max(1, len(filtered))
    qa_idx = [i for i,c in enumerate(filtered) if kind(c) == "qa" and c.get("id") not in seed_ids]
    max_qa = int(max_qa_frac * total)
    if len(qa_idx) > max_qa:
        drop = set(qa_idx[max_qa:])
        filtered = [c for i,c in enumerate(filtered) if i not in drop]
    return filtered

# -------------------------
# OpenAI calls
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

def generate_assessment(topic: str, goal: str, idioma: str, nivel: str, tipos: list, digest: str, max_qa_frac: float, require_anchor: bool, domain_kws: List[str]) -> Dict[str, Any]:
    user_payload = {
        "topic": topic, "goal": goal, "language": idioma, "level": nivel,
        "allowed_types": tipos, "materials_digest_excerpt": (digest or "")[:6000],
        "domain_keywords": domain_kws[:30], "max_qa_pct": max_qa_frac, "require_anchor": require_anchor
    }
    resp = chat_json(
        [{"role":"system","content":ASSESSMENT_SYSTEM},
         {"role":"user","content":json.dumps(user_payload, ensure_ascii=False)}],
        model=TEXT_MODEL, temperature=0
    )
    data = _safe_json(resp.choices[0].message.content or "{}")
    data.setdefault("summary","")
    data.setdefault("key_issues",[])
    data.setdefault("assumptions",[])
    data.setdefault("card_type_strategy",{"basic":35,"reverse":10,"cloze":35,"scenario":10,"procedure":10})
    data.setdefault("anchoring_plan","Reference file + page/section when available.")
    data.setdefault("terminology_focus",domain_kws[:15])
    data.setdefault("open_questions",[])
    data.setdefault("risks",[])
    data.setdefault("success_criteria",["Atomic cards","Anchored to materials","Variety beyond Q&A"])
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
    stable_model_id("Anki-Assistant Basic", version=3), "Anki-Assistant Basic (v3)",
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
    stable_model_id("Anki-Assistant Reverse", version=3), "Anki-Assistant Reverse (v3)",
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
    stable_model_id("Anki-Assistant Cloze", version=3), "Anki-Assistant Cloze (v3)",
    fields=[{"name":"Text"},{"name":"BackExtra"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Cloze Card",
        "qfmt":"<div class='hdr'>Q:</div><div class='front'>{{cloze:Text}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"<div class='hdr'>A:</div><div class='back'>{{BackExtra}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div class='audio'>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
               "<hr><div class='hdr'>Q:</div><div class='front'>{{cloze:Text}}</div>"
    }],
    css=COMMON_CSS,
    model_type=genanki.Model.CLOZE
)

def build_deck_id(title: str) -> int:
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return int(h[:10], 16)

def choose_tts_text(card: Dict[str, Any], policy: str, lang: str, front_raw: str, back_raw: str) -> Optional[str]:
    if policy == "none": return None
    audio_script = (card.get("audio_script") or "").strip()
    ex = ""
    if isinstance(card.get("examples"), list) and card["examples"]:
        ex = (card["examples"][0].get("text") or "").strip()
    p = policy.lower()
    candidates = []
    if p == "all":       candidates = [audio_script, ex, back_raw, front_raw]
    elif p == "answers": candidates = [audio_script, back_raw, ex, front_raw]
    elif p == "examples":candidates = [ex, audio_script, back_raw, front_raw]
    else:                candidates = [audio_script, back_raw, ex, front_raw]
    for c in candidates:
        t = strip_html_to_plain(c)
        if t: return t
    return None

def build_apkg_bytes(deck_json: Dict[str, Any], tts_policy: str = "examples", tts_coverage: str = "sampled", default_tag: str = "") -> bytes:
    meta = deck_json.get("deck", {})
    cards = deck_json.get("cards", [])
    title = meta.get("title","Anki Deck")
    lang  = meta.get("language","en")

    def anchored(c): return 1 if anchored_to_source(c) else 0
    def diff_rank(c): return {"easy":0,"medium":1,"hard":2}.get((c.get("difficulty") or "").lower(), 1)
    cards = sorted(cards, key=lambda c: (-anchored(c), c.get("type",""), diff_rank(c), c.get("id","")))

    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    # Who gets audio?
    idxs_for_audio = set()
    if tts_policy != "none" and len(cards) > 0:
        if tts_coverage.lower().startswith("all"):
            idxs_for_audio = set(range(len(cards)))
        else:
            idxs_for_audio = set(random.sample(range(len(cards)), min(MAX_AUDIO_FILES, len(cards))))

    # Pre-synthesize audio in parallel
    audio_map = {}
    if idxs_for_audio:
        def synth_one(idx):
            c = cards[idx]; fr, br = orient_q_a(c, lang)
            txt = choose_tts_text(c, tts_policy, lang, fr, br)
            if not txt: return idx, "", ""
            b = synth_audio(txt, lang)
            if not b: return idx, "", ""
            p = os.path.join(tmpdir, f"tts_{int(time.time()*1000)}_{idx}.mp3")
            with open(p, "wb") as f: f.write(b)
            return idx, f"[sound:{os.path.basename(p)}]", p
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(synth_one, i) for i in idxs_for_audio]
            for fut in as_completed(futs):
                try:
                    i, field, path = fut.result()
                    if field:
                        audio_map[i] = field
                        media_files.append(path)
                except Exception:
                    pass

    def add_note(c: Dict[str, Any], index: int):
        ctype = (c.get("type") or "basic").lower()
        hint = html_escape(c.get("hint") or "")
        examples_html = examples_to_html(c.get("examples"))
        src = c.get("source_ref") or {}
        extra_bits = []
        if (src.get("file") or src.get("page_or_time")):
            extra_bits.append(f"<div><b>Source:</b> {html_escape(src.get('file') or '')} {html_escape(src.get('page_or_time') or '')}</div>")
        extra = "".join(extra_bits)
        audio_field = audio_map.get(index, "")
        front_raw, back_raw = orient_q_a(c, lang)
        front = normalize_text_for_html(front_raw, cloze=(ctype=="cloze"))
        back  = normalize_text_for_html(back_raw)
        tags = sanitize_tags(c.get("tags", []))
        if default_tag: tags = sanitize_tags(tags + [default_tag])

        if ctype == "cloze":
            note = genanki.Note(model=MODEL_CLOZE,
                                fields=[front, back, hint, examples_html, audio_field, extra],
                                tags=tags)
        elif ctype == "reverse":
            note = genanki.Note(model=MODEL_REVERSE,
                                fields=[front, back, hint, examples_html, audio_field, extra],
                                tags=tags)
        else:
            note = genanki.Note(model=MODEL_BASIC,
                                fields=[front, back, hint, examples_html, audio_field, extra],
                                tags=tags)
        deck.add_note(note)

    for i, c in enumerate(cards):
        if isinstance(c, dict): add_note(c, i)

    pkg = genanki.Package(deck)
    if media_files: pkg.media_files = media_files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as tmp:
        pkg.write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            apkg_bytes = f.read()
    return apkg_bytes

# -------------------------
# Preview & stats
# -------------------------
def deck_stats(cards, domain_kws=None):
    def kind(c):
        t = (c.get("type") or "").lower()
        if t == "cloze": return "cloze"
        f = (c.get("front") or "").lower()
        if "?" in f: return "qa"
        if any(k in f for k in ["scenario:", "cenário:", "case:", "caso:"]): return "scenario"
        if any(k in f for k in ["procedure","procedimento","step","passo","ordem"]): return "procedure"
        return "basic"
    kinds = [kind(c) for c in cards]
    total = len(cards) or 1
    cover = 0
    if domain_kws:
        blob = " ".join(strip_html_to_plain((c.get("front",""))+" "+(c.get("back",""))) for c in cards).lower()
        cover = 100*sum(1 for k in domain_kws[:20] if k in blob)/max(1, min(20, len(domain_kws)))
    anch = 100*sum(1 for c in cards if anchored_to_source(c))/total
    return {
      "total": total,
      "qa_pct": 100*sum(1 for k in kinds if k=="qa")/total,
      "cloze_pct": 100*sum(1 for k in kinds if k=="cloze")/total,
      "scenario_pct": 100*sum(1 for k in kinds if k=="scenario")/total,
      "procedure_pct": 100*sum(1 for k in kinds if k=="procedure")/total,
      "avg_back_len": sum(len(strip_html_to_plain(c.get("back",""))) for c in cards)/total,
      "anchored_pct": anch,
      "coverage_pct": cover
    }

def render_card_preview(c: Dict[str, Any], lang: str) -> str:
    typ = (c.get("type") or "basic").title()
    front_raw, back_raw = orient_q_a(c, lang)
    front = normalize_text_for_html(front_raw, cloze=(typ.lower()=="cloze"))
    back  = normalize_text_for_html(back_raw)
    hint = html_escape(c.get("hint") or "")
    src = c.get("source_ref") or {}
    src_str = " • ".join([x for x in [src.get("file"), src.get("page_or_time")] if x])
    exs = examples_to_html(c.get("examples"))
    parts = [
        "<div class='card-prev'>",
        f"<div style='font-weight:600;opacity:.8;margin-bottom:6px;'>{typ}</div>",
        f"<div><b>Q:</b> {front}</div>",
        f"<div style='margin-top:6px;'><b>A:</b> {back}</div>",
    ]
    if hint: parts.append(f"<div style='margin-top:6px;color:#666;'><b>Hint:</b> {hint}</div>")
    if exs:  parts.append(f"<div style='margin-top:6px;'><b>Examples:</b> {exs}</div>")
    if src_str: parts.append(f"<div style='margin-top:6px;color:#444;'><b>Source:</b> {html_escape(src_str)}</div>")
    parts.append("</div>")
    return "".join(parts)

# -------------------------
# Conversation state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = ""
if "deck_title" not in st.session_state:
    st.session_state.deck_title = ""
if "materials" not in st.session_state:
    st.session_state.materials = []
if "digest" not in st.session_state:
    st.session_state.digest = ""
if "chosen_sections_meta" not in st.session_state:
    st.session_state.chosen_sections_meta = []
if "assessment" not in st.session_state:
    st.session_state.assessment = None
if "assessment_ok" not in st.session_state:
    st.session_state.assessment_ok = False
if "preview_cards" not in st.session_state:
    st.session_state.preview_cards = []
if "approved_cards" not in st.session_state:
    st.session_state.approved_cards = []
if "domain_kws" not in st.session_state:
    st.session_state.domain_kws = []

# -------------------------
# Sidebar — tools / settings
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings & Tools")
    st.caption("Explainable controls with safe defaults.")

    st.session_state.topic = st.text_area("Topic / Goal", st.session_state.topic or "French past tenses: usage & contrasts", help="Tell me what you want to learn. I’ll propose a plan.")
    st.session_state.deck_title = st.text_input("Deck title", st.session_state.deck_title or st.session_state.topic[:48] or "Anki Deck")

    idioma = st.selectbox("Deck language", ["fr-FR","pt-BR","en-US","es-ES","zh-CN"], index=0)
    nivel  = st.text_input("Level (optional)", "B1")
    goal   = st.selectbox("Focus", ["General learning","Org policy mastery","Exam prep","Language: Vocabulary","Language: Grammar & Patterns"], index=3)
    tipos  = st.multiselect("Allowed types", ["basic","reverse","cloze"], default=["basic","reverse","cloze"])
    target_n = st.slider("Target #cards", 6, 200, 36, 2)
    max_qa_pct = st.slider("Max % Q&A", 10, 90, 45, 5)
    require_anchor = st.checkbox("Require anchoring to sources (recommended for policies)", True)
    default_tag = st.text_input("Default tag (optional)", slugify(st.session_state.topic))

    st.markdown("---")
    st.caption("Attach materials (for better, anchored cards):")
    urls_text = st.text_area("URLs (one per line)", "", height=80)
    uploads = st.file_uploader("Files (PDF/DOCX/TXT/MD)", type=["pdf","docx","txt","md","markdown"], accept_multiple_files=True)
    colA, colB = st.columns(2)
    if colA.button("Add materials"):
        mats = ingest_files(uploads) if uploads else []
        if urls_text:
            urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
            mats += ingest_urls(urls)
        st.session_state.materials.extend(mats)
        st.success(f"Added {len(mats)} sources.")
    if colB.button("Clear materials"):
        st.session_state.materials = []
        st.session_state.digest = ""
        st.session_state.chosen_sections_meta = []
        st.session_state.assessment = None
        st.session_state.assessment_ok = False
        st.session_state.preview_cards = []
        st.session_state.approved_cards = []
        st.success("Cleared.")

    use_rag = st.checkbox("Use smart context (RAG)", True, help="Selects the most relevant sections from uploads/URLs.")
    rag_topk = st.slider("RAG sections (k)", 3, 12, 6)

    st.markdown("---")
    tts_mode = st.selectbox("TTS", ["none","answers","examples","all"], index=2)
    tts_cov  = st.selectbox("TTS coverage", ["Sampled (up to 24)","All"], index=0)

# Update digest if materials present
if st.session_state.materials:
    if use_rag:
        st.session_state.digest, st.session_state.chosen_sections_meta = rag_digest(
            st.session_state.materials, st.session_state.topic, "", top_k=rag_topk
        )
    else:
        st.session_state.digest = compress_materials_simple(st.session_state.materials)
    st.session_state.domain_kws = [w for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-_/\.]{4,}", st.session_state.digest.lower())][:40]

# -------------------------
# Chat area
# -------------------------
st.title("🧠 Anki Deck Assistant")
st.caption("Tell me what you need. I’ll propose a plan, show a preview, iterate, and export your .apkg.")

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Helper: add assistant/user messages
def say(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)

# Initial assistant nudge
if not st.session_state.messages:
    say("assistant",
        "Hi! I’m your deck-building assistant. Describe your learning goal/topic and, if you have them, attach documents/URLs in the sidebar. "
        "I’ll analyze, propose a plan, show a preview, and then we’ll build the final deck. 😊")

# Chat input
user_text = st.chat_input("Type your request or refinement…")
if user_text:
    say("user", user_text)

    # If we don’t have a plan yet (or user asked to 'plan'), propose one
    if (st.session_state.assessment is None) or re.search(r"\b(plan|assess|propose)\b", user_text.lower()):
        topic = st.session_state.topic or user_text
        deck_title = st.session_state.deck_title or topic[:48] or "Anki Deck"
        digest = st.session_state.digest
        domain_kws = st.session_state.domain_kws or []

        a = generate_assessment(topic, goal, idioma, nivel, tipos, digest, max_qa_pct/100.0, require_anchor, domain_kws)
        st.session_state.assessment = a
        plan_text = [
            f"**Proposed plan for _{deck_title}_**",
            f"**Summary:** {a.get('summary','')}",
            "**Key issues:**",
            *[f"- {x}" for x in a.get("key_issues",[])[:10]],
            "**Card type strategy (%):**",
            f"`{json.dumps(a.get('card_type_strategy',{}))}`",
            f"**Anchoring:** {a.get('anchoring_plan','')}",
            "**Open questions:**",
            *[f"- {x}" for x in a.get("open_questions",[]) or ["(none)"]],
            "_Use the buttons below to approve or ask for changes._"
        ]
        say("assistant", "\n".join(plan_text))

# Action bar under chat (proposal controls)
if st.session_state.assessment:
    c1, c2, c3 = st.columns([1,1,2])
    approve = c1.button("✅ Approve plan")
    refine  = c2.button("✍️ Suggest changes")
    preview_n = c3.selectbox("Preview size", [6,8,10,12,16], index=1)

    if refine:
        say("assistant", "Sure—tell me what to adjust (scope, card types, anchoring, difficulty, examples). I’ll update the plan or the preview accordingly.")
    if approve:
        st.session_state.assessment_ok = True
        say("assistant", "Great! Plan approved. I’m ready to draft a preview. Click **Generate preview** below when you’re set.")

# Preview controls
st.markdown("---")
colP1, colP2 = st.columns([1,3])
gen_prev = colP1.button("🔎 Generate preview")
regen_prev = colP1.button("🔁 Regenerate preview")
sample_size = colP1.selectbox("Cards to preview", [6,8,10,12,16], index=1)

# Build payload helper
def make_payload(n_cards: int, seed_cards=None):
    effective_tts = "none" if tts_mode == "none" else tts_mode
    return {
        "deck_title": st.session_state.deck_title or "Anki Deck",
        "idioma_alvo": idioma,
        "nivel_proficiencia": nivel,
        "topico": st.session_state.topic or "(no topic)",
        "limite_cartoes": n_cards,
        "tipos_permitidos": tipos,
        "politica_voz": f"tts={effective_tts}",
        "materiais_digest": st.session_state.digest,
        "extra_policy": "usage_tip",
        "goal": goal,
        "max_qa_pct": max_qa_pct/100.0,
        "require_anchor": require_anchor,
        "domain_keywords": st.session_state.domain_kws[:30],
        "user_feedback": "",  # chat refinements are implicit
        "seed_cards": seed_cards or [],
        "assessment_plan": st.session_state.assessment or {},
        "assessment_approved": bool(st.session_state.assessment_ok),
    }

def build_sample(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = gerar_baralho(payload)
    cards = dedupe_cards(data.get("cards", []))
    seed_ids = {c.get("id") for c in payload.get("seed_cards") or [] if c.get("id")}
    cards = enforce_variety(cards, st.session_state.domain_kws, payload["max_qa_pct"], payload["require_anchor"], seed_ids)
    data["cards"] = cards[: payload["limite_cartoes"]]
    data["deck"]["card_count_planned"] = len(data["cards"])
    return data

if gen_prev or regen_prev:
    if not (st.session_state.topic or "").strip():
        say("assistant", "Please add a topic in the sidebar first.")
    else:
        with st.spinner("Drafting preview…"):
            payload = make_payload(sample_size)
            data = build_sample(payload)
            st.session_state.preview_cards = data.get("cards", [])
            stats = deck_stats(st.session_state.preview_cards, st.session_state.domain_kws)
            say("assistant",
                f"Here’s a preview ({len(st.session_state.preview_cards)} cards). "
                f"Mix — Q&A {stats['qa_pct']:.0f}%, Cloze {stats['cloze_pct']:.0f}%, "
                f"Scenario {stats['scenario_pct']:.0f}%, Procedure {stats['procedure_pct']:.0f}%. "
                f"Anchored {stats['anchored_pct']:.0f}% • Keyword coverage {stats['coverage_pct']:.0f}%.")

# Show preview with optional edits
if st.session_state.preview_cards:
    deck_lang = idioma
    cols = st.columns(2)
    for i, c in enumerate(st.session_state.preview_cards):
        html = render_card_preview(c, deck_lang)
        (cols[i % 2]).markdown(html, unsafe_allow_html=True)

    if pd:
        st.markdown("**Edit / Approve preview cards** *(kept rows will seed the final deck)*")
        def cards_to_df(cards):
            rows = []
            for c in cards:
                rows.append({
                    "keep": True,
                    "type": c.get("type","basic"),
                    "front": strip_html_to_plain(c.get("front") or c.get("Text","")),
                    "back":  strip_html_to_plain(c.get("back") or c.get("BackExtra","")),
                    "hint":  c.get("hint",""),
                    "tags":  ",".join(sanitize_tags(c.get("tags",[])))
                })
            return pd.DataFrame(rows)
        def df_to_cards(df):
            out = []
            for _,r in df.iterrows():
                if not r.get("keep",False): continue
                tags = sanitize_tags([t.strip() for t in str(r.get("tags","")).split(",") if t.strip()])
                cid = f"seed-{hashlib.md5((str(r['type'])+str(r['front'])).encode()).hexdigest()[:8]}"
                out.append({
                  "id": cid,
                  "type": r["type"],
                  "front": normalize_text_for_html(r["front"], cloze=(r["type"]=="cloze")),
                  "back":  normalize_text_for_html(r["back"]),
                  "hint":  html_escape(r.get("hint","")),
                  "examples": [],
                  "tags": tags,
                  "source_ref": {}, "rationale": ""
                })
            return out

        df = cards_to_df(st.session_state.preview_cards)
        edited = st.data_editor(df, use_container_width=True, num_rows="fixed", key="preview_editor")
        st.session_state.approved_cards = df_to_cards(edited)

# -------------------------
# Build & export
# -------------------------
st.markdown("---")
colB1, colB2 = st.columns([1,3])
build_btn = colB1.button("🏗️ Build final deck (.apkg)")
if build_btn:
    if not (st.session_state.topic or "").strip():
        say("assistant", "Please add a topic first.")
    else:
        with st.spinner("Building your Anki package…"):
            # If the user approved edits, use them as seeds and ask for the remaining up to target_n
            seeds = st.session_state.approved_cards or []
            remain = max(0, target_n - len(seeds))
            payload = make_payload(remain or 0, seed_cards=seeds)
            if remain > 0:
                data = gerar_baralho(payload)
                more = dedupe_cards(data.get("cards", []))
                seed_ids = {c.get("id") for c in seeds if c.get("id")}
                more = enforce_variety(more, st.session_state.domain_kws, payload["max_qa_pct"], payload["require_anchor"], seed_ids)
                cards = dedupe_cards(seeds + more)[:target_n]
                deck_json = {"deck":{
                                "title": st.session_state.deck_title,
                                "language": idioma,
                                "level": nivel,
                                "topic": st.session_state.topic,
                                "source_summary": ""},
                             "cards": cards}
            else:
                deck_json = {"deck":{
                                "title": st.session_state.deck_title,
                                "language": idioma,
                                "level": nivel,
                                "topic": st.session_state.topic,
                                "source_summary": ""},
                             "cards": seeds[:target_n]}

            # Build package
            apkg = build_apkg_bytes(
                deck_json,
                tts_policy=tts_mode,
                tts_coverage=("all" if tts_cov.startswith("All") else "sampled"),
                default_tag=default_tag
            )
            fname = f"{slugify(st.session_state.deck_title)}_{int(time.time())}.apkg"
            st.success(f"Done! {len(deck_json['cards'])} cards generated.")
            st.download_button("⬇️ Download .apkg", data=apkg, file_name=fname, mime="application/octet-stream")

            # Quick stats
            stats = deck_stats(deck_json["cards"], st.session_state.domain_kws)
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            for k, title, val in [
                (k1,"Cards", f"{stats['total']}"),
                (k2,"Q&A %", f"{stats['qa_pct']:.0f}%"),
                (k3,"Cloze %", f"{stats['cloze_pct']:.0f}%"),
                (k4,"Scenario %", f"{stats['scenario_pct']:.0f}%"),
                (k5,"Anchored %", f"{stats['anchored_pct']:.0f}%"),
                (k6,"Coverage", f"{stats['coverage_pct']:.0f}%"),
            ]:
                with k:
                    st.markdown(f"<div class='kpi'><h3>{title}</h3><div class='val'>{val}</div></div>", unsafe_allow_html=True)

