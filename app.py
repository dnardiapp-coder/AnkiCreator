# app.py — Chat-First Anki Deck Creator (UX-polished & robust)

import os, io, json, time, tempfile, hashlib, re, random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from openai import OpenAI
import genanki
from gtts import gTTS
from gtts.lang import tts_langs
from pypdf import PdfReader
import docx as docxlib
from unidecode import unidecode

# ---- Optional dependencies (gracefully degraded) ----
try:
    from pytube import YouTube
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_DEPS_OK = True
except Exception:
    YOUTUBE_DEPS_OK = False
    YouTube = None
    YouTubeTranscriptApi = None

try:
    import requests
    from bs4 import BeautifulSoup
    try:
        from readability import Document
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
hr { border:none; border-top:1px solid #eee; margin:1rem 0; }
.step { padding:.5rem .75rem; border-radius:8px; border:1px solid #eaeaf2; background:#fafbff; margin-bottom:.5rem;}
.step .title { font-weight:600; }
.badge { display:inline-block; padding:.15rem .5rem; border-radius:999px; font-size:.75rem; background:#eef2ff; border:1px solid #e0e7ff; color:#3730a3; }
.help { color:#555; font-size:.92rem; }
.source-chip { display:inline-block; margin:2px 6px 2px 0; padding:.2rem .5rem; background:#f3f4f6; border:1px solid #e5e7eb; border-radius:999px; font-size:.8rem; }
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
# System prompts
# -------------------------
SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva que atua como Criador de Baralhos Anki.
Trabalhe em 3 passos: (1) entender pedido e propor um plano, (2) gerar prévia variada e específica, (3) produzir o baralho final.

Regras para os cartões:
- Recordação ativa e conhecimento atômico (uma ideia por cartão).
- Perguntas específicas, resposta única, linguagem clara, concisa.
- Variedade além de Q&A: cloze (um único alvo), cenários/mini-casos, procedimentos/passos, contrastes/exceções.
- Para políticas/exames: prazos, responsáveis, thresholds, exceções, auditoria, conformidade.
- Para idiomas: vocabulário (collocations), gramática/padrões (contraste), listening/pronúncia (script natural/IPA quando útil).
- Quando houver materiais (payload.materials_digest), ancore cartões neles e preencha source_ref.file/page_or_time.
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
Os percentuais devem somar aproximadamente 100 (ignore scenario/procedure se não aplicável).
"""


# -------------------------
# Session state
# -------------------------
def ss_init(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

for k, v in [
    ("messages", []),
    ("topic", ""),
    ("deck_title", ""),
    ("goal", "Language: Vocabulary"),
    ("idioma", "fr-FR"),
    ("nivel", "B1"),
    ("tipos", ["basic", "reverse", "cloze"]),
    ("target_n", 36),
    ("max_qa_pct", 0.45),
    ("require_anchor", True),
    ("default_tag", ""),
    ("materials", []),  # [{"file": ..., "content": ...}]
    ("urls_text", ""),
    ("youtube_urls_text", ""),
    ("digest", ""),
    ("chosen_sections_meta", []),
    ("domain_kws", []),
    ("assessment", None),
    ("assessment_ok", False),
    ("preview_cards", []),
    ("approved_cards", []),
    ("tts_mode", "examples"),
    ("tts_cov", "sampled"),
    ("mode", "Chat"),  # or "Form"
    ("pending_preview", False),
    ("pending_build", False),
]:
    ss_init(k, v)


# -------------------------
# Ingestion utilities
# -------------------------
def extract_pdf(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes); tmp.flush()
        reader = PdfReader(tmp.name)
        out: List[str] = []
        for i, p in enumerate(reader.pages, start=1):
            t = p.extract_text() or ""
            if t.strip():
                out.append(f"[p.{i}]\n{t}")
        return "\n\n".join(out)

def extract_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes); tmp.flush()
        doc = docxlib.Document(tmp.name)
        return "\n".join(p.text for p in doc.paragraphs)

def extract_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def ingest_files(uploaded_files) -> List[Dict[str, str]]:
    mats: List[Dict[str, str]] = []
    for f in (uploaded_files or []):
        name = f.name
        data = f.read()
        low = name.lower()
        if low.endswith(".pdf"):
            text = extract_pdf(data)
        elif low.endswith(".docx"):
            text = extract_docx(data)
        elif low.endswith((".txt", ".md", ".markdown")):
            text = extract_txt(data)
        else:
            continue
        mats.append({"file": name, "content": text})
    return mats

def fetch_url_text(url: str, timeout: int = 25) -> str:
    if not requests:
        return ""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "anki-deck-creator/1.0"})
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
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return soup.get_text("\n", strip=True)
        return html
    except Exception:
        return ""

def ingest_urls(urls: List[str]) -> List[Dict[str, str]]:
    mats: List[Dict[str, str]] = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        txt = fetch_url_text(u)
        if txt:
            name = re.sub(r"^https?://", "", u)[:80]
            mats.append({"file": name, "content": txt})
    return mats

def ingest_youtube_transcript(url: str) -> Optional[List[Dict[str, str]]]:
    if not YOUTUBE_DEPS_OK:
        return None
    try:
        yt = YouTube(url)
        video_id = yt.video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for lang in ['en', 'pt', 'fr']:
            try:
                transcript = transcript_list.find_transcript([lang])
                raw_text = transcript.fetch()
                full_transcript = " ".join([t['text'] for t in raw_text])
                title = yt.title
                return [{"file": f"youtube_{video_id}", "content": f"# {title}\n{full_transcript}"}]
            except Exception:
                continue
        return None
    except Exception as e:
        st.error(f"Error ingesting YouTube video: {e}")
        return None


# -------------------------
# RAG helpers
# -------------------------
def split_sections(text: str, file_name: str) -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    parts = re.split(r"\n(?=#+\s)|\n(?=\[p\.\d+\])", text)
    for i, part in enumerate(parts):
        pt = part.strip()
        if not pt:
            continue
        m = re.match(r"^(#+\s.*)", pt)
        if m:
            title = m.group(1)
        elif pt.lower().startswith("[p."):
            title = pt.split("\n", 1)[0]
        else:
            title = f"{file_name} — sec {i+1}"
        sections.append({"file": file_name, "title": title, "content": pt[:4000]})
    return sections or [{"file": file_name, "title": f"{file_name} — full", "content": text[:4000]}]

def simple_keyword_rank(query: str, sections: List[Dict[str, str]], top_k: int = 6) -> List[int]:
    q_toks = set(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]{3,}", query.lower()))
    scores: List[Tuple[int, int]] = []
    for i, s in enumerate(sections):
        txt = s["content"].lower()
        score = sum(1 for t in q_toks if t in txt)
        scores.append((score, i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:top_k]]

def tfidf_rank(query: str, sections: List[Dict[str, str]], top_k: int = 6) -> List[int]:
    docs = [s["content"] for s in sections]
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.9)
        X = vec.fit_transform(docs)
        qv = vec.transform([query])
        sims = cosine_similarity(qv, X).ravel()
        top = sims.argsort()[::-1][:top_k]
        return top.tolist()
    except Exception:
        return simple_keyword_rank(query, sections, top_k)

def rag_digest(materials: List[Dict[str, str]], topic: str, user_feedback: str,
               top_k: int = 6, max_chars: int = 12000) -> Tuple[str, List[Dict[str, str]]]:
    if not materials:
        return "", []
    sections: List[Dict[str, str]] = []
    for m in materials:
        sections.extend(split_sections(m["content"], m["file"]))
    query = f"{topic}\n{user_feedback}".strip()
    idxs = tfidf_rank(query, sections, top_k) if SKLEARN_OK else simple_keyword_rank(query, sections, top_k)
    chosen = [sections[i] for i in idxs]
    parts: List[str] = []
    for s in chosen:
        hdr = f"# {s['file']} — {s['title']}"
        parts.append(hdr + "\n" + s["content"])
    digest = "\n\n".join(parts)
    return digest[:max_chars], chosen


# -------------------------
# LLM helpers
# -------------------------
def chat_json(messages, model=TEXT_MODEL, temperature=0, max_tries=4):
    last_err = None
    for i in range(max_tries):
        try:
            with st.spinner("Calling AI Model..."):
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
            return resp
        except Exception as e:
            last_err = e
            time.sleep(min(8, 2 ** i + random.random()))
    raise last_err

def _safe_json(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        return {}


# -------------------------
# Validation & variety
# -------------------------
REQUIRED_CARD_FIELDS = {"type", "front", "back"}

def strip_html_to_plain(s: str) -> str:
    if not s:
        return ""
    s2 = re.sub(r"<[^>]+>", " ", s)
    s2 = (s2.replace("&nbsp;", " ").replace("&amp;", "&")
           .replace("&lt;", "<").replace("&gt;", ">"))
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2

def valid_card(c: dict) -> bool:
    if not isinstance(c, dict):
        return False
    # accept normalized ('front'/'back') or raw cloze 'Text'
    typ = (c.get("type") or "").lower()
    if typ == "cloze":
        cloze_src = c.get("front", "") or c.get("Text", "")
        if "{{c" not in cloze_src:
            return False
        # ensure 'back' exists (may be placeholder); keep short answer explanation
        if len(strip_html_to_plain(c.get("back", ""))) > 420:
            return False
        return True
    # basic/reverse
    if not all(k in c and (c[k] is not None and len(str(c[k]).strip()) > 0) for k in REQUIRED_CARD_FIELDS):
        return False
    if len(strip_html_to_plain(c.get("back", ""))) > 420:
        return False
    return True

def dedupe_cards(cards: list) -> list:
    def sig(c):
        t = (c.get("type") or "").lower().strip()
        f = (c.get("front") or c.get("Text") or "").strip().lower()
        b = (c.get("back") or c.get("BackExtra") or "").strip().lower()
        return hashlib.md5(f"{t}|{f[:160]}|{b[:160]}".encode()).hexdigest()
    seen, out = set(), []
    for c in (cards or []):
        if not isinstance(c, dict):
            continue
        s = sig(c)
        if s in seen:
            continue
        seen.add(s); out.append(c)
    return out

def contains_domain_keyword(card: dict, kws: List[str]) -> bool:
    if not kws:
        return True
    blob = f"{card.get('front','')} | {card.get('back','')}".lower()
    return any(k in blob for k in kws[:40])

def kind_of(c: dict) -> str:
    t = (c.get("type") or "").lower()
    if t == "cloze":
        return "cloze"
    f = (c.get("front") or "").lower()
    if "?" in f:
        return "qa"
    if any(k in f for k in ["scenario:", "cenário:", "case:", "caso:"]):
        return "scenario"
    if any(k in f for k in ["procedure", "procedimento", "step", "passo", "ordem"]):
        return "procedure"
    return "basic"

def anchored_to_source(card: dict) -> bool:
    src = card.get("source_ref") or {}
    return bool(src.get("file") or src.get("page_or_time"))

def enforce_variety(cards: List[dict], kws: List[str], max_qa_frac: float,
                    require_anchor_effective: bool, seed_ids: set) -> List[dict]:
    filtered: List[dict] = []
    for c in cards:
        if require_anchor_effective and not anchored_to_source(c) and c.get("id") not in seed_ids:
            continue
        if not contains_domain_keyword(c, kws) and c.get("id") not in seed_ids:
            continue
        filtered.append(c)
    if not filtered:
        # fallback to preserve UX (avoid empty preview)
        return cards
    total = max(1, len(filtered))
    qa_idx = [i for i, c in enumerate(filtered) if kind_of(c) == "qa" and c.get("id") not in seed_ids]
    max_qa = int(max_qa_frac * total)
    if len(qa_idx) > max_qa:
        drop = set(qa_idx[max_qa:])
        filtered = [c for i, c in enumerate(filtered) if i not in drop]
    return filtered


# -------------------------
# Text & audio helpers
# -------------------------
def slugify(text: str, maxlen: int = 64) -> str:
    t = unidecode(text or "").lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t[:maxlen] or f"slug-{int(time.time())}"

def html_escape(s: Optional[str]) -> str:
    if not s:
        return ""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def normalize_text_for_html(s: str, cloze: bool = False) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^```+|\n```+$", "", s)
    if cloze:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    else:
        s = html_escape(s)
    s = s.replace("\n", "<br>")
    return s

def examples_to_html(examples: Optional[List[Dict[str, str]]]) -> str:
    if not examples:
        return ""
    items: List[str] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        line = html_escape(ex.get("text", ""))
        tr = ex.get("translation")
        nt = ex.get("notes")
        if tr:
            line += f' <span class="tr">({html_escape(tr)})</span>'
        if nt:
            line += f' <span class="nt">[{html_escape(nt)}]</span>'
        items.append(f"<li>{line}</li>")
    return "<ul class='examples-list'>" + "".join(items) + "</ul>"

def map_lang_for_gtts(language_code: str) -> str:
    langs = tts_langs()
    if not language_code:
        return "en"
    lc = language_code.lower()
    prefs = {
        "fr": ["fr"], "pt": ["pt", "pt-br"], "en": ["en", "en-us", "en-gb"],
        "es": ["es"], "de": ["de"], "it": ["it"],
        "zh": ["zh", "zh-cn", "cmn-cn", "zh-tw"], "ja": ["ja"],
    }
    for k, choices in prefs.items():
        if lc.startswith(k):
            for code in choices:
                if code in langs:
                    return code
    for cand in (lc, lc.replace("_", "-")):
        if cand in langs:
            return cand
    return "en"

def synth_audio(text: str, lang_code: str) -> Optional[bytes]:
    t = (text or "").strip()
    if not t or len(t) > AUDIO_CHAR_LIMIT:
        return None
    try:
        fp = io.BytesIO()
        gTTS(text=t, lang=map_lang_for_gtts(lang_code)).write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None

def _clean_tag(tag) -> str:
    t = str(tag or "").strip().lower()
    t = re.sub(r"\s+", "-", t)
    t = re.sub(r"[^a-z0-9_\-:]", "", t)
    return t[:40]

def sanitize_tags(tags) -> list:
    if not isinstance(tags, list):
        return []
    out, seen = [], set()
    for t in tags:
        ct = _clean_tag(t)
        if ct and ct not in seen:
            out.append(ct); seen.add(ct)
        if len(out) >= 12:
            break
    return out

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
    if not text:
        return False
    t = text.strip()
    if "?" in t:
        return True
    lp = _lang_prefix(lang)
    w = re.sub(r'^[\-\•\–\s]+', '', t.lower())
    return any(w.startswith(q + " ") or w == q for q in QUESTION_WORDS.get(lp, []))

def orient_q_a(card: Dict[str, Any], lang: str) -> Tuple[str, str]:
    f = (card.get("front") or card.get("Text") or "").strip()
    b = (card.get("back") or card.get("BackExtra") or "").strip()
    fq, bq = looks_like_question(f, lang), looks_like_question(b, lang)
    if fq and not bq: return f, b
    if bq and not fq: return b, f
    if len(f) > len(b): return b, f
    return f, b


# -------------------------
# LLM pipelines
# -------------------------
def gerar_baralho(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Gere um baralho Anki a partir do payload a seguir (JSON):\n" + json.dumps(payload, ensure_ascii=False)}
        ]
        resp = chat_json(messages, model=TEXT_MODEL, temperature=0)
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        data = _safe_json(content or "{}")
    except Exception as e:
        st.error(f"Failed to generate cards from AI model: {e}")
        return {"deck": {}, "cards": []}

    cards = data.get("cards", [])
    if not isinstance(cards, list):
        cards = []

    # --- Normalize BEFORE validation (handle cloze Text/BackExtra) ---
    normalized = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        typ = (c.get("type") or "basic").lower()
        if typ == "cloze":
            if not c.get("front") and c.get("Text"):
                c["front"] = str(c.get("Text", ""))
            if not c.get("back"):
                be = str(c.get("BackExtra", "")).strip()
                c["back"] = be if be else "(cloze answer on reveal)"
        c["front"] = str(c.get("front", "")).strip()
        c["back"]  = str(c.get("back",  "")).strip()
        normalized.append(c)
    cards = normalized
    # ---------------------------------------------------------------

    # Validate and normalize HTML
    cards = [c for c in cards if valid_card(c)]
    for c in cards:
        typ = (c.get("type") or "basic").lower()
        if typ == "cloze":
            c["front"] = normalize_text_for_html(c.get("front", ""), cloze=True)
            c["back"] = normalize_text_for_html(c.get("back", ""))
        else:
            c["front"] = normalize_text_for_html(c.get("front", ""))
            c["back"] = normalize_text_for_html(c.get("back", ""))

    data.setdefault("deck", {})
    data["deck"].setdefault("title", payload.get("deck_title", "Anki Deck"))
    data["deck"].setdefault("language", payload.get("idioma_alvo", "en"))
    data["deck"].setdefault("level", payload.get("nivel_proficiencia", ""))
    data["deck"].setdefault("topic", payload.get("topico", ""))
    data["deck"]["card_count_planned"] = len(cards)
    data["cards"] = cards
    return data

def generate_assessment(topic: str, goal: str, idioma: str, nivel: str,
                        tipos: list, digest: str, domain_kws: List[str],
                        max_qa_frac: float, require_anchor: bool) -> Dict[str, Any]:
    payload = {
        "topic": topic, "goal": goal, "language": idioma, "level": nivel,
        "allowed_types": tipos, "materials_digest_excerpt": (digest or "")[:6000],
        "domain_keywords": domain_kws[:30], "max_qa_pct": max_qa_frac, "require_anchor": require_anchor
    }
    try:
        messages = [
            {"role": "system", "content": ASSESSMENT_SYSTEM},
            {"role": "user", "content": "Analise o pedido/materiais e proponha um plano de baralho (JSON):\n" + json.dumps(payload, ensure_ascii=False)}
        ]
        resp = chat_json(messages, model=TEXT_MODEL, temperature=0)
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        data = _safe_json(content or "{}")
    except Exception as e:
        st.error(f"Failed to generate assessment: {e}")
        data = {}

    data.setdefault("summary", "")
    data.setdefault("key_issues", [])
    data.setdefault("card_type_strategy", {"basic": 35, "reverse": 10, "cloze": 35, "scenario": 10, "procedure": 10})
    data.setdefault("anchoring_plan", "Reference file + page/section when available. If no materials, rely on topic-only generation.")
    data.setdefault("terminology_focus", domain_kws[:12])
    data.setdefault("open_questions", [])
    return data


# -------------------------
# Anki models & deck builder
# -------------------------
COMMON_CSS = """
.card { font-family: -apple-system, Segoe UI, Roboto, Arial; font-size: 20px; text-align: left; color: #222; background: #fff; }
.front { font-size: 1.05em; line-height: 1.45; }
.back { line-height: 1.5; }
.hint { margin-top: 8px; font-size: 0.95em; color: #666; }
.examples-list { padding-left: 18px; }
.tr { color: #444; font-style: italic; }
.nt { color: #666; }
.extra { margin-top: 10px; font-size: 0.9em; color: #444; }
.audio { margin-top: 8px; }
hr { margin: 14px 0; }
"""

def stable_model_id(name: str, version: int = 3) -> int:
    h = hashlib.sha1(f"{name}-v{version}".encode("utf-8")).hexdigest()
    return int(h[:10], 16)

MODEL_BASIC = genanki.Model(
    stable_model_id("Anki-Chat Basic", version=3), "Anki-Chat Basic (v3)",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
        {"name": "Hint"},
        {"name": "Examples"},
        {"name": "Audio"},
        {"name": "Extra"},
    ],
    templates=[{
        "name": "Card 1",
        "qfmt": "<div class='front'>{{Front}}</div><div class='hint'>{{Hint}}</div>",
        "afmt": "{{FrontSide}}<hr id='answer'><div class='back'>{{Back}}</div><div class='examples'>{{Examples}}</div><div class='audio'>{{Audio}}</div><div class='extra'>{{Extra}}</div>",
    }],
    css=COMMON_CSS
)

MODEL_REVERSE = genanki.Model(
    stable_model_id("Anki-Chat Reverse", version=3), "Anki-Chat Reverse (v3)",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
        {"name": "Hint"},
        {"name": "Examples"},
        {"name": "Audio"},
        {"name": "Extra"},
    ],
    templates=[
        {
            "name": "Forward",
            "qfmt": "<div class='front'>{{Front}}</div><div class='hint'>{{Hint}}</div>",
            "afmt": "{{FrontSide}}<hr id='answer'><div class='back'>{{Back}}</div><div class='examples'>{{Examples}}</div><div class='audio'>{{Audio}}</div><div class='extra'>{{Extra}}</div>",
        },
        {
            "name": "Reverse",
            "qfmt": "<div class='front'>{{Back}}</div><div class='hint'>{{Hint}}</div>",
            "afmt": "{{FrontSide}}<hr id='answer'><div class='back'>{{Front}}</div><div class='examples'>{{Examples}}</div><div class='audio'>{{Audio}}</div><div class='extra'>{{Extra}}</div>",
        },
    ],
    css=COMMON_CSS
)

MODEL_CLOZE = genanki.Model(
    stable_model_id("Anki-Chat Cloze", version=3), "Anki-Chat Cloze (v3)",
    fields=[
        {"name": "Text"},
        {"name": "BackExtra"},
        {"name": "Hint"},
        {"name": "Examples"},
        {"name": "Audio"},
        {"name": "Extra"},
    ],
    templates=[{
        "name": "Cloze",
        "qfmt": "{{cloze:Text}}<div class='hint'>{{Hint}}</div>",
        "afmt": "{{cloze:Text}}<hr id='answer'><div class='back'>{{BackExtra}}</div><div class='examples'>{{Examples}}</div><div class='audio'>{{Audio}}</div><div class='extra'>{{Extra}}</div>",
    }],
    css=COMMON_CSS,
    model_type=genanki.Model.CLOZE
)

def build_deck_id(title: str) -> int:
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return int(h[:10], 16)

def choose_tts_text(card: Dict[str, Any], policy: str, lang: str, front_raw: str, back_raw: str) -> Optional[str]:
    if policy == "none":
        return None
    audio_script = (card.get("audio_script") or "").strip()
    ex = ""
    if isinstance(card.get("examples"), list) and card["examples"]:
        ex = (card["examples"][0].get("text") or "").strip()
    p = policy.lower()
    if p == "all":
        candidates = [audio_script, ex, back_raw, front_raw]
    elif p == "answers":
        candidates = [audio_script, back_raw, ex, front_raw]
    elif p == "examples":
        candidates = [ex, audio_script, back_raw, front_raw]
    else:
        candidates = [audio_script, back_raw, ex, front_raw]
    for c in candidates:
        t = strip_html_to_plain(c)
        if t:
            return t
    return None

def build_apkg_bytes(deck_json: Dict[str, Any], tts_policy: str = "examples",
                     tts_coverage: str = "sampled", default_tag: str = "") -> bytes:
    meta = deck_json.get("deck", {})
    cards = deck_json.get("cards", [])
    title = meta.get("title", "Anki Deck")
    lang = meta.get("language", "en")

    def anchored(c): return 1 if anchored_to_source(c) else 0
    def diff_rank(c): return {"easy": 0, "medium": 1, "hard": 2}.get((c.get("difficulty") or "").lower(), 1)

    cards = sorted(cards, key=lambda c: (-anchored(c), c.get("type", ""), diff_rank(c), c.get("id", "")))
    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    # TTS selection
    idxs_for_audio = set()
    if tts_policy != "none" and len(cards) > 0:
        if tts_coverage.lower().startswith("all"):
            idxs_for_audio = set(range(len(cards)))
        else:
            idxs_for_audio = set(random.sample(range(len(cards)), min(MAX_AUDIO_FILES, len(cards))))
    audio_map: Dict[int, str] = {}

    if idxs_for_audio:
        def synth_one(idx):
            c = cards[idx]; fr, br = orient_q_a(c, lang)
            txt = choose_tts_text(c, tts_policy, lang, fr, br)
            if not txt:
                return idx, "", ""
            b = synth_audio(txt, lang)
            if not b:
                return idx, "", ""
            p = os.path.join(tmpdir, f"tts_{int(time.time()*1000)}_{idx}.mp3")
            with open(p, "wb") as f:
                f.write(b)
            return idx, f"[sound:{os.path.basename(p)}]", p

        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(synth_one, i) for i in idxs_for_audio]
            with st.spinner("Generating audio..."):
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
        extra_bits: List[str] = []
        if (src.get("file") or src.get("page_or_time")):
            extra_bits.append(f"<div><b>Source:</b> {html_escape(src.get('file') or '')} {html_escape(src.get('page_or_time') or '')}</div>")
        extra = "".join(extra_bits)
        audio_field = audio_map.get(index, "")
        front_raw, back_raw = orient_q_a(c, lang)
        front = normalize_text_for_html(front_raw, cloze=(ctype == "cloze"))
        back = normalize_text_for_html(back_raw)
        tags = sanitize_tags(c.get("tags", []))
        if default_tag:
            tags = sanitize_tags(tags + [default_tag])
        guid = genanki.guid_for(front, back)

        if ctype == "cloze":
            note = genanki.Note(model=MODEL_CLOZE,
                                fields=[front, back, hint, examples_html, audio_field, extra],
                                tags=tags, guid=guid)
        elif ctype == "reverse":
            note = genanki.Note(model=MODEL_REVERSE,
                                fields=[front, back, hint, examples_html, audio_field, extra],
                                tags=tags, guid=guid)
        else:
            note = genanki.Note(model=MODEL_BASIC,
                                fields=[front, back, hint, examples_html, audio_field, extra],
                                tags=tags, guid=guid)
        deck.add_note(note)

    progress_bar = st.progress(0)
    for i, c in enumerate(cards):
        if isinstance(c, dict):
            add_note(c, i)
        progress_bar.progress(int(((i + 1) / max(1, len(cards))) * 100))
    progress_bar.empty()

    pkg = genanki.Package(deck)
    if media_files:
        pkg.media_files = media_files

    with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as tmp:
        pkg.write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            apkg_bytes = f.read()
    return apkg_bytes


# -------------------------
# Stats & rendering
# -------------------------
def deck_stats(cards, domain_kws=None):
    kinds = [kind_of(c) for c in cards]
    total = len(cards) or 1
    cover = 0
    if domain_kws:
        blob = " ".join(strip_html_to_plain((c.get("front", "")) + " " + (c.get("back", ""))) for c in cards).lower()
        cover = 100 * sum(1 for k in domain_kws[:20] if k in blob) / max(1, min(20, len(domain_kws)))
    anch = 100 * sum(1 for c in cards if anchored_to_source(c)) / total
    return {
        "total": total,
        "qa_pct": 100 * sum(1 for k in kinds if k == "qa") / total,
        "cloze_pct": 100 * sum(1 for k in kinds if k == "cloze") / total,
        "scenario_pct": 100 * sum(1 for k in kinds if k == "scenario") / total,
        "procedure_pct": 100 * sum(1 for k in kinds if k == "procedure") / total,
        "avg_back_len": sum(len(strip_html_to_plain(c.get("back", ""))) for c in cards) / total,
        "anchored_pct": anch,
        "coverage_pct": cover
    }

def render_card_preview(c: Dict[str, Any], lang: str) -> str:
    typ = (c.get("type") or "basic").title()
    front_raw, back_raw = orient_q_a(c, lang)
    front = normalize_text_for_html(front_raw, cloze=(typ.lower() == "cloze"))
    back = normalize_text_for_html(back_raw)
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
    if hint:
        parts.append(f"<div style='margin-top:6px;color:#666;'><b>Hint:</b> {hint}</div>")
    if exs:
        parts.append(f"<div style='margin-top:6px;'><b>Examples:</b> {exs}</div>")
    if src_str:
        parts.append(f"<div style='margin-top:6px;color:#444;'><b>Source:</b> {html_escape(src_str)}</div>")
    parts.append("</div>")
    return "".join(parts)


# -------------------------
# UX helpers
# -------------------------
def stepper():
    st.markdown("<div class='step'><span class='title'>1) Plan</span> <span class='badge'>Assessment</span><br><span class='help'>Set topic/goal/language. (Optional) Add sources to anchor.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='step'><span class='title'>2) Preview</span> <span class='badge'>Sample cards</span><br><span class='help'>Review & edit a small sample. Approved rows become seed cards.</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='step'><span class='title'>3) Build</span> <span class='badge'>.apkg export</span><br><span class='help'>Generate full deck with optional TTS and download.</span></div>", unsafe_allow_html=True)

def apply_chat_commands(user_text: str):
    # lightweight command parsing from chat
    m = re.search(r"\bset size\s+(\d{1,3})\b", user_text.lower())
    if m:
        st.session_state["target_n"] = max(4, min(200, int(m.group(1))))
        st.toast(f"Target size set to {st.session_state['target_n']} cards.", icon="✅")
    m = re.search(r"\blang(uage)?\s+([A-Za-z\-]{2,8})", user_text.lower())
    if m:
        st.session_state["idioma"] = m.group(2)
        st.toast(f"Deck language set to {st.session_state['idioma']}.", icon="✅")
    if re.search(r"\b(goal|mode)\b.*(policy|exam|vocab|grammar)", user_text.lower()):
        if "policy" in user_text.lower(): st.session_state["goal"] = "Org policy mastery"
        elif "exam" in user_text.lower(): st.session_state["goal"] = "Exam prep"
        elif "vocab" in user_text.lower(): st.session_state["goal"] = "Language: Vocabulary"
        elif "grammar" in user_text.lower(): st.session_state["goal"] = "Language: Grammar & Patterns"
        st.toast(f"Goal set to {st.session_state['goal']}.", icon="✅")

def require_anchor_effective() -> bool:
    return bool(st.session_state.get("require_anchor", True) and (st.session_state.get("materials") or st.session_state.get("digest")))

def make_payload(n_cards: int, seed_cards=None):
    return {
        "deck_title": st.session_state.get("deck_title") or "Anki Deck",
        "idioma_alvo": st.session_state.get("idioma", "en"),
        "nivel_proficiencia": st.session_state.get("nivel", ""),
        "topico": st.session_state.get("topic") or "(no topic)",
        "limite_cartoes": n_cards,
        "tipos_permitidos": st.session_state.get("tipos", ["basic","reverse","cloze"]),
        "politica_voz": f"tts={st.session_state.get('tts_mode','examples')}",
        "materials_digest": st.session_state.get("digest",""),
        "extra_policy": "usage_tip",
        "goal": st.session_state.get("goal","Language: Vocabulary"),
        "max_qa_pct": st.session_state.get("max_qa_pct", 0.45),
        "require_anchor": require_anchor_effective(),
        "domain_keywords": (st.session_state.get("domain_kws") or [])[:30],
        "user_feedback": "",
        "seed_cards": seed_cards or [],
        "assessment_plan": st.session_state.get("assessment") or {},
        "assessment_approved": bool(st.session_state.get("assessment_ok")),
    }

def build_sample(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = gerar_baralho(payload)
    cards = dedupe_cards(data.get("cards", []))
    seed_ids = {c.get("id") for c in payload.get("seed_cards", []) if c.get("id")}
    cards = enforce_variety(cards, st.session_state.get("domain_kws", []), payload["max_qa_pct"], payload["require_anchor"], seed_ids)
    data["cards"] = (cards or [])[: payload["limite_cartoes"]]
    data["deck"]["card_count_planned"] = len(data["cards"])
    return data

def reset_all():
    for k in list(st.session_state.keys()):
        if k not in ["mode"]:  # keep mode
            del st.session_state[k]
    # reinit
    for k, v in [
        ("messages", []),
        ("topic", ""),
        ("deck_title", ""),
        ("goal", "Language: Vocabulary"),
        ("idioma", "fr-FR"),
        ("nivel", "B1"),
        ("tipos", ["basic", "reverse", "cloze"]),
        ("target_n", 36),
        ("max_qa_pct", 0.45),
        ("require_anchor", True),
        ("default_tag", ""),
        ("materials", []),
        ("urls_text", ""),
        ("youtube_urls_text", ""),
        ("digest", ""),
        ("chosen_sections_meta", []),
        ("domain_kws", []),
        ("assessment", None),
        ("assessment_ok", False),
        ("preview_cards", []),
        ("approved_cards", []),
        ("tts_mode", "examples"),
        ("tts_cov", "sampled"),
        ("pending_preview", False),
        ("pending_build", False),
    ]:
        ss_init(k, v)


# -------------------------
# Layout
# -------------------------
header_l, header_r = st.columns([3,2])
with header_l:
    st.title("🧠 Chat Anki Deck Creator")
    st.caption("Plan → Preview → Build high-quality Anki decks from topics or documents. Anchors to your sources when provided.")
with header_r:
    with st.container():
        cols = st.columns(3)
        with cols[0]:
            st.metric("Target", st.session_state["target_n"])
        with cols[1]:
            st.metric("Max Q&A %", f"{int(st.session_state['max_qa_pct']*100)}%")
        with cols[2]:
            st.metric("Mode", st.session_state["mode"])

top_cols = st.columns([1,1,2,1])
with top_cols[0]:
    st.session_state["mode"] = st.radio("Mode", ["Chat","Form"], horizontal=True, index=["Chat","Form"].index(st.session_state["mode"]))
with top_cols[1]:
    if st.button("🔄 Reset session"):
        reset_all()
        st.rerun()
with top_cols[2]:
    st.write("")
    st.caption("Tip: In chat you can say “**language en-US**”, “**set size 80**”, “**preview**”, or “**build**”.")
with top_cols[3]:
    pass

st.divider()
left, right = st.columns([2,1])

with right:
    st.subheader("📎 Sources (optional)")
    with st.expander("Add / Manage Sources", expanded=False):
        upl = st.file_uploader("Upload files", type=["pdf","docx","txt","md","markdown"], accept_multiple_files=True)
        url_lines = st.text_area("URLs (one per line)", st.session_state["urls_text"], height=80)
        youtube_urls = st.text_area("YouTube URLs (one per line)", st.session_state["youtube_urls_text"], height=80)
        c1, c2 = st.columns(2)
        if c1.button("Add sources"):
            with st.spinner("Processing sources..."):
                mats = ingest_files(upl) if upl else []
                if url_lines.strip():
                    st.session_state["urls_text"] = url_lines
                    urls = [u.strip() for u in url_lines.splitlines() if u.strip()]
                    mats += ingest_urls(urls)
                if youtube_urls.strip() and YOUTUBE_DEPS_OK:
                    st.session_state["youtube_urls_text"] = youtube_urls
                    yt_urls = [u.strip() for u in youtube_urls.splitlines() if u.strip()]
                    for u in yt_urls:
                        yt_mat = ingest_youtube_transcript(u) or []
                        mats += yt_mat
                if mats:
                    st.session_state["materials"].extend(mats)
                    st.toast(f"Added {len(mats)} new sources!", icon="✅")
                else:
                    st.toast("No new sources added.", icon="ℹ️")
        if c2.button("Clear sources"):
            st.session_state["materials"] = []
            st.session_state["digest"] = ""
            st.session_state["chosen_sections_meta"] = []
            st.toast("Sources cleared.", icon="🗑️")

    if st.session_state["materials"]:
        st.markdown("**Current sources**")
        for m in st.session_state["materials"]:
            st.markdown(f"<span class='source-chip'>{m['file']}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ Build Settings")
    st.session_state["tts_mode"] = st.selectbox("TTS", ["none","answers","examples","all"], index=["none","answers","examples","all"].index(st.session_state["tts_mode"]))
    st.session_state["tts_cov"] = st.selectbox("TTS coverage", ["sampled","all"], index=["sampled","all"].index(st.session_state["tts_cov"]))
    st.session_state["default_tag"] = st.text_input("Default tag", st.session_state["default_tag"] or slugify(st.session_state.get("topic","")))
    st.session_state["require_anchor"] = st.checkbox("Require anchoring (auto-disabled if no materials)", value=st.session_state["require_anchor"])
    st.session_state["max_qa_pct"] = st.slider("Max % Q&A", 10, 90, int(st.session_state["max_qa_pct"]*100), 5)/100.0
    st.session_state["target_n"] = st.slider("Target cards", 4, 200, st.session_state["target_n"], 2)

with left:
    stepper()
    st.subheader("1) Plan")
    plan_cols = st.columns(2)
    with plan_cols[0]:
        st.session_state["topic"] = st.text_input("Topic", st.session_state["topic"])
        st.session_state["deck_title"] = st.text_input("Deck title", st.session_state["deck_title"] or st.session_state["topic"][:48])
        st.session_state["idioma"] = st.text_input("Language code", st.session_state["idioma"])
    with plan_cols[1]:
        st.session_state["nivel"] = st.text_input("Level", st.session_state["nivel"])
        st.session_state["goal"] = st.selectbox("Goal", ["General learning","Org policy mastery","Exam prep","Language: Vocabulary","Language: Grammar & Patterns"],
                                                index=["General learning","Org policy mastery","Exam prep","Language: Vocabulary","Language: Grammar & Patterns"].index(st.session_state["goal"]))
        st.session_state["tipos"] = st.multiselect("Allowed types", ["basic","reverse","cloze"], default=st.session_state["tipos"])

    # Build digest/keywords if materials exist
    if st.session_state["materials"]:
        digest, chosen = rag_digest(st.session_state["materials"], st.session_state["topic"], "", top_k=6)
        st.session_state["digest"] = digest
        st.session_state["chosen_sections_meta"] = chosen
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-_/\.]{4,}", digest.lower())
        st.session_state["domain_kws"] = list(dict.fromkeys(toks))[:40]
    else:
        st.session_state["digest"] = ""
        toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-_/\.]{4,}", (st.session_state["topic"] or "").lower())
        st.session_state["domain_kws"] = list(dict.fromkeys(toks))[:20]

    plan_actions = st.columns([1,1,1,2])
    gen_plan = plan_actions[0].button("🧭 Generate plan")
    do_preview_btn = plan_actions[1].button("🔎 Preview")
    do_build_btn = plan_actions[2].button("🏗️ Build")

    if gen_plan:
        if not st.session_state["topic"].strip():
            st.warning("Add a topic first.")
        else:
            a = generate_assessment(
                st.session_state["topic"], st.session_state["goal"], st.session_state["idioma"], st.session_state["nivel"],
                st.session_state["tipos"], st.session_state["digest"], st.session_state["domain_kws"],
                st.session_state["max_qa_pct"], st.session_state["require_anchor"]
            )
            st.session_state["assessment"] = a
            st.session_state["assessment_ok"] = True
            st.success("Plan generated.")
    if st.session_state["assessment"]:
        a = st.session_state["assessment"]
        st.markdown("**Assessment plan**")
        st.json(a, expanded=False)

    st.markdown("---")
    st.subheader("2) Preview")
    prev_size = st.selectbox("Preview size", [1,2,3,4,5], index=1, help="Small sample to review/edit before full build.")

    if do_preview_btn:
        if not st.session_state["topic"].strip():
            st.warning("Add a topic first.")
        else:
            payload = make_payload(prev_size)
            data = build_sample(payload)
            st.session_state["preview_cards"] = data.get("cards", [])
            stats = deck_stats(st.session_state["preview_cards"], st.session_state["domain_kws"])
            st.info(f"Preview ready ({len(st.session_state['preview_cards'])} cards). "
                    f"Mix — Q&A {stats['qa_pct']:.0f}%, Cloze {stats['cloze_pct']:.0f}%, "
                    f"Scenario {stats['scenario_pct']:.0f}%, Procedure {stats['procedure_pct']:.0f}%. "
                    f"Anchored {stats['anchored_pct']:.0f}%.")

    if st.session_state["preview_cards"]:
        deck_lang = st.session_state["idioma"]
        cols = st.columns(2)
        for i, c in enumerate(st.session_state["preview_cards"]):
            html = render_card_preview(c, deck_lang)
            (cols[i % 2]).markdown(html, unsafe_allow_html=True)

        st.caption("Approve/edit the sample (kept rows become seed cards for the final deck).")
        if pd:
            def cards_to_df(cards):
                rows = []
                for c in cards:
                    rows.append({
                        "keep": True,
                        "type": c.get("type","basic"),
                        "front": strip_html_to_plain(c.get("front") or c.get("Text","")),
                        "back":  strip_html_to_plain(c.get("back") or c.get("BackExtra","")),
                        "hint": c.get("hint",""),
                        "tags": ",".join(sanitize_tags(c.get("tags",[])))
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
            df = cards_to_df(st.session_state["preview_cards"])
            edited = st.data_editor(df, use_container_width=True, num_rows="fixed", key="preview_editor")
            st.session_state["approved_cards"] = df_to_cards(edited)
        else:
            # Minimal fallback if pandas isn't installed
            keep_options = [st.checkbox(f"Keep card {i+1}", value=True, key=f"keep_{i}") for i in range(len(st.session_state["preview_cards"]))]
            approved = []
            for i, c in enumerate(st.session_state["preview_cards"]):
                if keep_options[i]:
                    approved.append(c)
            st.session_state["approved_cards"] = approved

    st.markdown("---")
    st.subheader("3) Build")
    if do_build_btn:
        if not st.session_state["topic"].strip():
            st.warning("Add a topic first.")
        else:
            with st.spinner("Building your deck…"):
                seeds = st.session_state["approved_cards"] or []
                remain = max(0, st.session_state["target_n"] - len(seeds))
                if remain > 0:
                    payload = make_payload(remain, seed_cards=seeds)
                    data = gerar_baralho(payload)
                    more = dedupe_cards(data.get("cards", []))
                    seed_ids = {c.get("id") for c in seeds if c.get("id")}
                    more = enforce_variety(more, st.session_state["domain_kws"], payload["max_qa_pct"], payload["require_anchor"], seed_ids)
                    cards = dedupe_cards(seeds + (more or []))[: st.session_state["target_n"]]
                else:
                    cards = seeds[: st.session_state["target_n"]]

                deck_json = {
                    "deck": {
                        "title": st.session_state["deck_title"] or st.session_state["topic"][:48] or "Anki Deck",
                        "language": st.session_state["idioma"],
                        "level": st.session_state["nivel"],
                        "topic": st.session_state["topic"],
                        "source_summary": ""
                    },
                    "cards": cards
                }
                apkg = build_apkg_bytes(
                    deck_json,
                    tts_policy=st.session_state["tts_mode"],
                    tts_coverage=("all" if st.session_state["tts_cov"].startswith("all") else "sampled"),
                    default_tag=(st.session_state["default_tag"] or slugify(st.session_state["topic"]))
                )
                fname = f"{slugify(deck_json['deck']['title'])}_{int(time.time())}.apkg"
                st.success(f"Deck built with {len(deck_json['cards'])} cards!")
                st.download_button("⬇️ Download .apkg", data=apkg, file_name=fname, mime="application/octet-stream")
                stats = deck_stats(deck_json["cards"], st.session_state["domain_kws"])
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                for k, title, val in [
                    (k1, "Cards", f"{stats['total']}"),
                    (k2, "Q&A %", f"{stats['qa_pct']:.0f}%"),
                    (k3, "Cloze %", f"{stats['cloze_pct']:.0f}%"),
                    (k4, "Scenario %", f"{stats['scenario_pct']:.0f}%"),
                    (k5, "Anchored %", f"{stats['anchored_pct']:.0f}%"),
                    (k6, "Coverage", f"{stats['coverage_pct']:.0f}%"),
                ]:
                    with k:
                        st.markdown(f"<div class='kpi'><h3>{title}</h3><div class='val'>{val}</div></div>", unsafe_allow_html=True)

# -------------------------
# Chat panel (unified with the same pipeline)
# -------------------------
st.divider()
st.subheader("💬 Chat (optional, same flow)")
if not st.session_state["messages"]:
    st.session_state["messages"] = [{"role":"assistant","content":"Tell me your learning goal or topic (e.g., *French past tenses*, *Org security policy basics*, *Exam prep: CISSP Domain 1*). I’ll propose a plan, then show a preview."}]

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("Say: 'French past tenses', or 'language en-US', or 'preview', or 'build'…")
if user_text:
    st.session_state["messages"].append({"role":"user","content":user_text})
    apply_chat_commands(user_text)

    # lightweight intents
    asked_preview = bool(re.search(r"\b(preview|show sample|sample)\b", user_text.lower()))
    asked_build = bool(re.search(r"\b(build|make deck|generate deck|export)\b", user_text.lower()))
    asked_plan = bool(re.search(r"\b(plan|assessment)\b", user_text.lower()))

    # auto-set topic if empty
    if not st.session_state["topic"].strip():
        st.session_state["topic"] = user_text.strip()
        st.session_state["deck_title"] = st.session_state["deck_title"] or st.session_state["topic"][:48]

    reply_chunks: List[str] = []

    if asked_plan or (st.session_state["assessment"] is None and st.session_state["topic"].strip()):
        a = generate_assessment(
            st.session_state["topic"], st.session_state["goal"], st.session_state["idioma"], st.session_state["nivel"],
            st.session_state["tipos"], st.session_state["digest"], st.session_state["domain_kws"],
            st.session_state["max_qa_pct"], st.session_state["require_anchor"]
        )
        st.session_state["assessment"] = a
        st.session_state["assessment_ok"] = True
        reply_chunks.append("Plan generated. Use **preview** to see sample cards, or **build** to export the deck.")

    if asked_preview:
        payload = make_payload(3)
        data = build_sample(payload)
        st.session_state["preview_cards"] = data.get("cards", [])
        reply_chunks.append(f"Preview ready: {len(st.session_state['preview_cards'])} card(s).")

    if asked_build:
        seeds = st.session_state["approved_cards"] or st.session_state["preview_cards"] or []
        remain = max(0, st.session_state["target_n"] - len(seeds))
        if remain > 0:
            payload = make_payload(remain, seed_cards=seeds)
            data = gerar_baralho(payload)
            more = dedupe_cards(data.get("cards", []))
            seed_ids = {c.get("id") for c in seeds if c.get("id")}
            more = enforce_variety(more, st.session_state["domain_kws"], payload["max_qa_pct"], payload["require_anchor"], seed_ids)
            cards = dedupe_cards(seeds + (more or []))[: st.session_state["target_n"]]
        else:
            cards = seeds[: st.session_state["target_n"]]
        deck_json = {
            "deck": {
                "title": st.session_state["deck_title"] or st.session_state["topic"][:48] or "Anki Deck",
                "language": st.session_state["idioma"],
                "level": st.session_state["nivel"],
                "topic": st.session_state["topic"],
                "source_summary": ""
            },
            "cards": cards
        }
        apkg = build_apkg_bytes(
            deck_json,
            tts_policy=st.session_state["tts_mode"],
            tts_coverage=("all" if st.session_state["tts_cov"].startswith("all") else "sampled"),
            default_tag=(st.session_state["default_tag"] or slugify(st.session_state["topic"]))
        )
        fname = f"{slugify(deck_json['deck']['title'])}_{int(time.time())}.apkg"
        st.session_state["__last_pkg"] = (apkg, fname)
        reply_chunks.append(f"Deck built with {len(deck_json['cards'])} cards. Scroll up to the Build section to download.")

    if not reply_chunks:
        reply_chunks.append("Got it. You can say **plan**, **preview**, or **build** — or adjust topic/goal above.")

    st.session_state["messages"].append({"role":"assistant","content":"\n\n".join(reply_chunks)})
    st.rerun()
