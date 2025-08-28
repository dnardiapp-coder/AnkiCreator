# app.py — Anki-Generator (Pro UI)
# Beautiful, explainable UI • Preview + Edit • RAG optional • Variety controls
# Parallel TTS • Auto-split long answers • Coverage diagnostics • CSV import/export

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
st.set_page_config(page_title="🧠 Anki-Generator", page_icon="🧠", layout="wide")

# Subtle CSS polish + Reader Mode hook
BASE_CSS = """
<style>
:root { --acc:#5b8df7; }
.block-container { padding-top: 1.2rem; }
.small-note { font-size:0.88rem; color:#666; }
.help-badge { display:inline-block; background:#eef3ff; color:#2d3e72; padding:3px 8px; border-radius:9px; font-size:0.8rem; margin-left:6px; }
hr { border:none; border-top:1px solid #eee; margin:1rem 0; }
.kpi { border:1px solid #eee; border-radius:12px; padding:10px; text-align:center; }
.kpi h3 { margin:0; font-size:0.9rem; color:#555; }
.kpi .val { font-weight:700; font-size:1.2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 6px 12px; border-radius: 8px; background: #f7f7fb; }
.reader .stMarkdown, .reader .stText { font-size: 1.07rem; line-height: 1.55; }
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# -------------------------
# API key & client
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Please set OPENAI_API_KEY in Streamlit Secrets or as an environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

# -------------------------
# Constants
# -------------------------
TEXT_MODEL = "gpt-4o-mini"
STRICT_MAX_ROUNDS_DEFAULT = 4
STRICT_BATCH_DEFAULT = 20
STRICT_HARD_TIMEOUT = 180
MAX_AUDIO_FILES = 24
AUDIO_CHAR_LIMIT = 400

# -------------------------
# Prompt (with Flashcard Guide + feedback)
# -------------------------
SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva, integrada ao 'Anki-Generator'.

OBJETIVO
- Gerar cartões úteis, específicos e de alta qualidade, ancorados no tópico e nos materiais do usuário (quando houver).
- Práticas baseadas em evidência: recordação ativa, conhecimento atômico, exemplos concretos, interleaving/variação, contraste/erro comum, foco em transferência para uso real.
- Considere o feedback do usuário (payload.user_feedback) para ajustar foco, granularidade, estilo, exemplos, tipos e cobertura.

FINS (payload.goal)
- "General learning": mix equilibrado (definição, cloze, cenários, procedimentos).
- "Org policy mastery": foque seções/IDs, prazos, thresholds, exceções, responsáveis, aprovações, sanções, conformidade, auditoria, SLA/OLA; muitos cenários e procedimentos.
- "Exam prep": pontos cobrados/pegadinhas/variações; >=30% cloze, >=20% cenários; inclua erros comuns.
- "Language: Vocabulary": collocations, regência/partículas, falsos cognatos; cloze de palavras; exemplos bilíngues; audio_script apropriado.
- "Language: Grammar & Patterns": contraste de padrões; cloze morfossintático; condições/restrições; exemplos mínimos; audio_script com frases-alvo.
- "Language: Listening-Pronunciation": priorize áudio, IPA quando aplicável; cloze auditiva (palavra/chunk); script natural.
- "Language: Reading-CEFR": cloze de conectores/tempos/referência pronominal; microtrechos com inferência.

MIX & VARIEDADE
- Tipos: basic | reverse | cloze (evite múltipla escolha).
- Varie conteúdos:
  • Definição/critério diagnóstico + exemplo.
  • Cloze (omita termo/número/data/cláusula/passo).
  • Cenário/mini-caso → decisão/aplicação da regra/política.
  • Procedimento/checklist (ordem, responsáveis, prazos).
  • Contraste/exceções/thresholds (se/então, valores, datas).
- Respeite payload.max_qa_pct (no máx. Q&A diretos); favoreça cloze/cenário/procedimento conforme meta.

ESPECIFICIDADE & ANCORAGEM
- Use payload.materiais_digest quando existir. Ao derivar do material, preencha source_ref.file e page_or_time.
- Utilize payload.domain_keywords nos enunciados/respostas.
- rationale segue payload.extra_policy (usage_tip/common_pitfall/mnemonic/self_check/source/none).

GUIA DE FLASHCARDS (20 regras, resumido)
1) Uma informação por cartão. 2) Perguntas específicas. 3) Linguagem clara/simples.
4) Resposta única. 5) Cartões concisos. 6) Duas vias quando fizer sentido.
7) Active recall (Q/cloze). 8) 2–5s para responder. 9) Foco em conhecimento estável.
10) Conceitos precisos. 11) Escreva com suas palavras. 12) Exemplos p/ ideias abstratas.
13) Pistas mínimas (hint curto). 14) Camadas (básico→complexo). 15) Quebre assuntos grandes.
16) Não confunda reconhecimento com recall. 17) Cloze com 1 alvo. 18) Refine cartões ruins.
19) Personalize aos objetivos. 20) Teste para uso real.

SAÍDA (JSON)
{
 "deck":{"title":"string","language":"string","level":"string","topic":"string","source_summary":"string","card_count_planned":number},
 "cards":[
   {"id":"string","type":"basic|reverse|cloze","front":"string","back":"string","hint":"string|null",
    "examples":[{"text":"string","translation":"string|null","notes":"string|null"}],
    "language_fields":{"ipa":"string|null","pinyin":"string|null","morphology":"string|null","register":"string|null"},
    "audio_script":"string|null","tags":["string"],"difficulty":"easy|medium|hard",
    "source_ref":{"file":"string|null","page_or_time":"string|null","span":"string|null"},
    "rationale":"string"}],
 "qa_report":{"duplicates_removed":number,"too_broad_removed":number,"notes":"string"}
}
- Faça 'card_count_planned' = len(cards).
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

# Q/A heuristics
QUESTION_WORDS = {
    "fr": ["qui","que","quoi","quel","quelle","quels","quelles","où","quand","comment","pourquoi","combien"],
    "pt": ["o que","qual","quais","quando","onde","como","por que","por quê","quem","quanto","quantos","quantas"],
    "en": ["what","which","when","where","how","why","who","whom","whose"],
    "es": ["qué","cuál","cuáles","cuándo","dónde","cómo","por qué","quién","quiénes","cuánto","cuántos","cuántas"],
    "de": ["was","welcher","welche","welches","wann","wo","wie","warum","wer","wessen","wem","wieviel"],
    "it": ["che","quale","quali","quando","dove","come","perché","chi","quanto","quanti","quante"],
}
def _lang_prefix(lang: str) -> str:
    l = (lang or "en").lower()
    for k in ["fr","pt","es","de","it","en"]:
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

def extra_label(kind: str, lang: str) -> str:
    lp = (lang or "en").lower()
    if lp.startswith("fr"):
        return {"usage_tip":"Astuce d’usage :", "common_pitfall":"Piège courant :", "mnemonic":"Mnémotechnique :", "self_check":"Auto-vérification :", "source":"Source :", "none":""}.get(kind,"")
    if lp.startswith("pt"):
        return {"usage_tip":"Dica de uso:", "common_pitfall":"Erro comum:", "mnemonic":"Mnemônico:", "self_check":"Auto-checagem:", "source":"Fonte:", "none":""}.get(kind,"")
    if lp.startswith("es"):
        return {"usage_tip":"Consejo de uso:", "common_pitfall":"Error común:", "mnemonic":"Mnemotecnia:", "self_check":"Autoevaluación:", "source":"Fuente:", "none":""}.get(kind,"")
    if lp.startswith("de"):
        return {"usage_tip":"Praxistipp:", "common_pitfall":"Häufige Falle:", "mnemonic":"Eselsbrücke:", "self_check":"Selbsttest:", "source":"Quelle:", "none":""}.get(kind,"")
    if lp.startswith("it"):
        return {"usage_tip":"Suggerimento d’uso:", "common_pitfall":"Errore comune:", "mnemonic":"Mnemotecnica:", "self_check":"Auto-verifica:", "source":"Fonte:", "none":""}.get(kind,"")
    return {"usage_tip":"Usage tip:", "common_pitfall":"Common pitfall:", "mnemonic":"Mnemonic:", "self_check":"Self-check:", "source":"Source:", "none":""}.get(kind,"")

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
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"anki-generator/1.0"})
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
    # cap per-section to keep context tight
    per = max(800, max_chars // max(1, len(materials)))
    parts = []
    for m in materials:
        chunk = (m["content"] or "")[:per]
        parts.append(f"# {m['file']}\n{chunk}")
    return "\n\n".join(parts)[:max_chars]

# -------------------------
# Mix minima
# -------------------------
def goal_mix_minima(goal: str) -> Dict[str, float]:
    g = (goal or "").lower()
    if "policy" in g: return {"cloze": 0.25, "scenario": 0.35, "procedure": 0.25}
    if "exam prep" in g: return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}
    if "vocabulary" in g: return {"cloze": 0.35, "scenario": 0.05, "procedure": 0.00}
    if "grammar" in g or "patterns" in g: return {"cloze": 0.50, "scenario": 0.10, "procedure": 0.00}
    if "listening" in g or "pronunciation" in g: return {"cloze": 0.30, "scenario": 0.10, "procedure": 0.00}
    if "reading" in g or "cefr" in g: return {"cloze": 0.40, "scenario": 0.10, "procedure": 0.00}
    return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}

# -------------------------
# OpenAI wrapper + JSON
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

REQUIRED_CARD_FIELDS = {"type","front","back"}
def valid_card(c: dict) -> bool:
    if not isinstance(c, dict): return False
    typ = (c.get("type") or "").lower()
    if typ == "cloze":
        c.setdefault("front", c.get("Text",""))
        c.setdefault("back",  c.get("BackExtra",""))
    if not all(k in c for k in REQUIRED_CARD_FIELDS): return False
    # one fact per card heuristic: answers not too long
    if len(strip_html_to_plain(c.get("back",""))) > 420:
        return False
    # reject cloze with multiple targets
    if typ == "cloze" and re.findall(r"\{\{c\d+::", c.get("front","")).count("{{") > 1:
        return False
    return True

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
        if not isinstance(c, dict): continue
        ctype = (c.get("type") or "basic").lower()
        if ctype == "cloze":
            if not c.get("front") and c.get("Text"): c["front"] = c["Text"]
            if not c.get("back")  and c.get("BackExtra"): c["back"] = c["BackExtra"]
            c["front"] = normalize_text_for_html(c.get("front",""), cloze=True)
            c["back"]  = normalize_text_for_html(c.get("back",""))
        else:
            c["front"] = normalize_text_for_html(c.get("front",""))
            c["back"]  = normalize_text_for_html(c.get("back",""))

    data.setdefault("deck", {})
    data["deck"].setdefault("title", payload.get("deck_title","Anki-Generator Deck"))
    data["deck"].setdefault("language", payload.get("idioma_alvo","en"))
    data["deck"].setdefault("level", payload.get("nivel_proficiencia",""))
    data["deck"].setdefault("topic", payload.get("topico",""))
    data["deck"]["card_count_planned"] = len(cards)
    data["cards"] = cards

    if hasattr(resp, "usage"):
        st.caption(f"Tokens — prompt: {resp.usage.prompt_tokens}, completion: {resp.usage.completion_tokens}")
    return data

# -------------------------
# Variety, anchoring & dedupe
# -------------------------
STOPWORDS = set("""
a an and the of de da do dos das para por com sem em no na nos nas um uma umas uns que quem como quando onde porque porquê se então or ou e not não ao aos às à pela pelo pelos pelas este esta isto esse essa isso aquele aquela aquilo entre sobre até desde contra sob além cada mais menos muito muita muitos muitas pouco pouca poucos poucas ser estar ter haver foi são eram será deverá deveráo dever deverá deverão pode podem não sim
""".split())

def extract_domain_keywords(text: str, top_n: int = 40) -> List[str]:
    if not text: return []
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9\-_/\.]{3,}", text.lower())
    freq = {}
    for t in toks:
        if t in STOPWORDS: continue
        if t.isdigit(): continue
        if len(t) < 4: continue
        freq[t] = freq.get(t, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def card_kind(card: dict) -> str:
    t = (card.get("type") or "").lower()
    if t == "cloze": return "cloze"
    f = (card.get("front") or "").lower()
    if "?" in f: return "qa"
    if any(k in f for k in ["cenário:", "scenario:", "case:", "caso:"]): return "scenario"
    if any(k in f for k in ["procedimento", "procedure", "passo", "step", "ordem", "sequence"]): return "procedure"
    return "basic"

def anchored_to_source(card: dict) -> bool:
    src = card.get("source_ref") or {}
    return bool(src.get("file") or src.get("page_or_time"))

def contains_domain_keyword(card: dict, kws: List[str]) -> bool:
    if not kws: return True
    blob = f"{card.get('front','')} || {card.get('back','')}".lower()
    return any(k in blob for k in kws[:40])

def jaccard_near_dup(a: str, b: str, threshold: float = 0.85) -> bool:
    ta=set(re.findall(r"\w+", strip_html_to_plain(a.lower())))
    tb=set(re.findall(r"\w+", strip_html_to_plain(b.lower())))
    if not ta or not tb: return False
    j = len(ta&tb)/max(1,len(ta|tb))
    return j > threshold

def _card_signature(card: dict) -> str:
    t = (card.get("type") or "").lower().strip()
    f = (card.get("front") or card.get("Text") or "").strip().lower()
    b = (card.get("back") or card.get("BackExtra") or "").strip().lower()
    return f"{t}|{f[:160]}|{b[:160]}"

def dedupe_cards(cards: list) -> list:
    seen, out = set(), []
    for c in cards or []:
        if not isinstance(c, dict): continue
        sig = _card_signature(c)
        if sig in seen: continue
        # near-dup
        is_nd = False
        for d in out[-50:]:
            if jaccard_near_dup((c.get("front",""))+(c.get("back","")),
                                (d.get("front",""))+(d.get("back",""))):
                is_nd = True; break
        if is_nd: continue
        seen.add(sig); out.append(c)
    return out

def enforce_mix_and_anchoring(cards: List[dict], kws: List[str], require_anchor: bool,
                              max_qa_frac: float, minima_pct: Dict[str,float],
                              seed_ids: Optional[set] = None) -> Tuple[List[dict], dict]:
    seed_ids = seed_ids or set()
    filtered = []
    for c in cards:
        if require_anchor and not anchored_to_source(c) and c.get("id") not in seed_ids:
            continue
        if not contains_domain_keyword(c, kws) and c.get("id") not in seed_ids:
            continue
        filtered.append(c)

    if not filtered:
        return [], {"cloze_min":0,"scenario_min":0,"procedure_min":0}

    total = max(1, len(filtered))
    qa_cards_idx = [i for i,c in enumerate(filtered) if card_kind(c)=="qa" and c.get("id") not in seed_ids]
    max_qa = int(max_qa_frac * total)
    if len(qa_cards_idx) > max_qa:
        qa_sorted = sorted(qa_cards_idx, key=lambda i: len((filtered[i].get("front",""))))
        to_remove = set(qa_sorted[max_qa:])
        filtered = [c for i,c in enumerate(filtered) if i not in to_remove]

    kinds = [card_kind(c) for c in filtered]
    need = {
        "cloze_min": max(0, int(minima_pct.get("cloze",0)*len(filtered)) - kinds.count("cloze")),
        "scenario_min": max(0, int(minima_pct.get("scenario",0)*len(filtered)) - kinds.count("scenario")),
        "procedure_min": max(0, int(minima_pct.get("procedure",0)*len(filtered)) - kinds.count("procedure")),
    }
    return filtered, need

# -------------------------
# Auto-split long answers → atomic cards
# -------------------------
def split_long_card(card: Dict[str,Any], lang: str, max_parts: int = 6) -> List[Dict[str,Any]]:
    back_plain = strip_html_to_plain(card.get("back",""))
    if len(back_plain) < 240:
        return [card]
    text = back_plain
    # Try split by numbered/newline bullets, else semicolons, else commas (last resort)
    parts = []
    if re.search(r"(\n|\r)", text):
        parts = [p.strip("-• ").strip() for p in re.split(r"[\n\r]+", text) if p.strip()]
    if not parts and ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
    if not parts and "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 2:
        return [card]
    parts = parts[:max_parts]
    front_raw, _ = orient_q_a(card, lang)
    new_cards = []
    for i, p in enumerate(parts, 1):
        nc = dict(card)
        nc["id"] = f"{card.get('id','auto')}-split-{i}"
        # Make the question targeted
        nc["front"] = normalize_text_for_html(f"{strip_html_to_plain(front_raw)} — item {i}/{len(parts)}")
        nc["back"] = normalize_text_for_html(p)
        # Mark as step/component
        tags = sanitize_tags((card.get("tags", []) or []) + ["component"])
        nc["tags"] = tags
        new_cards.append(nc)
    return new_cards

def autosplit_deck(cards: List[Dict[str,Any]], lang: str, cap_growth: int = 40) -> List[Dict[str,Any]]:
    out = []
    budget = cap_growth  # never explode the deck
    for c in cards:
        if budget <= 0:
            out.append(c); continue
        pieces = split_long_card(c, lang)
        if len(pieces) == 1:
            out.append(c)
        else:
            add_n = min(len(pieces), budget)
            out.extend(pieces[:add_n])
            budget -= add_n-1  # first replaces the original
    return out

# -------------------------
# Strict mode top-up
# -------------------------
def gerar_cartoes_adicionais(payload: dict, ja_gerados: list, faltantes: int, lote: int) -> list:
    resumo = []
    for c in ja_gerados or []:
        resumo.append({"type": c.get("type"),
                       "front": c.get("front") or c.get("Text"),
                       "back":  c.get("back")  or c.get("BackExtra"),
                       "tags":  c.get("tags", [])})
    pedir = min(max(0,int(faltantes)), int(lote))
    if pedir <= 0: return []

    sys_addendum = (
        SYSTEM_PROMPT +
        "\n\nINSTRUÇÃO ADICIONAL: Gere EXATAMENTE o número solicitado e responda SOMENTE com JSON no formato "
        '{"cards":[{...}]}' " (sem 'deck' e sem 'qa_report'). Não repita cartões existentes."
    )
    mix_targets = payload.get("mix_targets", {})
    sys_addendum += (
        f"\n\nPRIORIZAR NESTA RODADA (se aplicável): "
        f"cloze+={mix_targets.get('cloze_min',0)}, "
        f"cenários+={mix_targets.get('scenario_min',0)}, "
        f"procedimentos+={mix_targets.get('procedure_min',0)}. "
        "Respeite max Q&A (payload_base.max_qa_pct). "
        "Use payload_base.domain_keywords. "
        "Se require_anchor=true e houver materiais, preencha source_ref.file e page_or_time."
    )

    pedido = {
        "pedido": f"Gerar exatamente {pedir} cartões NÃO-DUPLICADOS, variados (cloze/cenário/procedimento/definição-criterial).",
        "payload_base": {
            "idioma_alvo": payload.get("idioma_alvo"),
            "nivel_proficiencia": payload.get("nivel_proficiencia"),
            "topico": payload.get("topico"),
            "tipos_permitidos": payload.get("tipos_permitidos"),
            "politica_voz": payload.get("politica_voz"),
            "materiais_digest": payload.get("materiais_digest",""),
            "extra_policy": payload.get("extra_policy","usage_tip"),
            "goal": payload.get("goal","General learning"),
            "domain_keywords": payload.get("domain_keywords", []),
            "require_anchor": payload.get("require_anchor", True),
            "max_qa_pct": payload.get("max_qa_pct", 0.5),
            "mix_targets": payload.get("mix_targets", {}),
            "user_feedback": payload.get("user_feedback","")
        },
        "cartoes_ja_gerados_resumo": resumo
    }

    resp = chat_json(
        [{"role":"system","content":sys_addendum},
         {"role":"user","content":json.dumps(pedido, ensure_ascii=False)}],
        model=TEXT_MODEL, temperature=0
    )
    data = _safe_json(resp.choices[0].message.content or "{}")
    novos = data.get("cards", [])
    if not isinstance(novos, list): novos = []
    novos = [c for c in novos if valid_card(c)]

    for c in novos:
        typ = (c.get("type") or "basic").lower()
        if typ == "cloze":
            if not c.get("front") and c.get("Text"): c["front"] = c["Text"]
            if not c.get("back")  and c.get("BackExtra"): c["back"] = c["BackExtra"]
            c["front"] = normalize_text_for_html(c.get("front",""), cloze=True)
            c["back"]  = normalize_text_for_html(c.get("back",""))
        else:
            c["front"] = normalize_text_for_html(c.get("front",""))
            c["back"]  = normalize_text_for_html(c.get("back",""))
    return novos[:pedir]

def gerar_baralho_estrito(payload: dict, progress=None, max_rounds: int = STRICT_MAX_ROUNDS_DEFAULT, batch_size: int = STRICT_BATCH_DEFAULT) -> dict:
    start = time.time()
    desired = int(payload.get("limite_cartoes", 20))
    seed_cards = payload.get("seed_cards", []) or []
    base = gerar_baralho(payload)
    cards = dedupe_cards(seed_cards + base.get("cards", []))

    kws = payload.get("domain_keywords", [])
    minima_pct = payload.get("minima_overrides") or goal_mix_minima(payload.get("goal","General learning"))
    seed_ids = {c.get("id") for c in seed_cards if c.get("id")}
    cards, need = enforce_mix_and_anchoring(cards, kws, payload.get("require_anchor", True),
                                            payload.get("max_qa_pct", 0.5), minima_pct, seed_ids)

    # Auto-split long answers (atomicity)
    deck_lang = (base.get("deck", {}).get("language") or payload.get("idioma_alvo","en"))
    cards = autosplit_deck(cards, deck_lang)

    if progress: progress.progress(min(0.1, len(cards)/max(1, desired)))
    rounds = 0

    while len(cards) < desired and rounds < max_rounds:
        if time.time() - start > STRICT_HARD_TIMEOUT: break
        faltam = desired - len(cards)
        payload["mix_targets"] = need
        novos = gerar_cartoes_adicionais(payload, cards, faltam, lote=batch_size)

        prev = len(cards)
        draft = dedupe_cards(cards + novos)
        draft, need = enforce_mix_and_anchoring(draft, kws, payload.get("require_anchor", True),
                                                payload.get("max_qa_pct", 0.5), minima_pct, seed_ids)
        draft = autosplit_deck(draft, deck_lang)
        cards = draft

        rounds += 1
        if progress:
            frac = min(0.95, len(cards) / max(1, desired) * 0.9 + 0.05)
            progress.progress(frac)
        if len(cards) == prev:
            break

    base["cards"] = cards[:desired]
    base.setdefault("deck", {})
    base["deck"]["title"] = payload.get("deck_title", base["deck"].get("title","Anki-Generator Deck"))
    base["deck"]["card_count_planned"] = len(base["cards"])
    if progress: progress.progress(1.0)
    return base

# -------------------------
# Anki models (v3)
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
    stable_model_id("Anki-Generator Basic", version=3), "Anki-Generator Basic (v3)",
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
    stable_model_id("Anki-Generator Reverse", version=3), "Anki-Generator Reverse (v3)",
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
    stable_model_id("Anki-Generator Cloze", version=3), "Anki-Generator Cloze (v3)",
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

# TTS selection & parallel synth
def choose_tts_text(card: Dict[str, Any], policy: str, lang: str, front_raw: str, back_raw: str) -> Optional[str]:
    if policy == "nenhuma": return None
    audio_script = (card.get("audio_script") or "").strip()
    ex = ""
    if isinstance(card.get("examples"), list) and card["examples"]:
        ex = (card["examples"][0].get("text") or "").strip()
    p = policy.lower()
    candidates = []
    if p == "todas":       candidates = [audio_script, ex, back_raw, front_raw]
    elif p == "respostas": candidates = [audio_script, back_raw, ex, front_raw]
    elif p == "exemplos":  candidates = [ex, audio_script, back_raw, front_raw]
    else:                  candidates = [audio_script, back_raw, ex, front_raw]
    for c in candidates:
        t = strip_html_to_plain(c)
        if t: return t
    return None

def build_apkg_bytes(
    deck_json: Dict[str, Any],
    tts_policy: str = "exemplos",
    extra_kind: str = "usage_tip",
    tts_coverage: str = "sampled",   # "sampled" | "all"
    default_tag: str = ""
) -> bytes:
    meta = deck_json.get("deck", {})
    cards = deck_json.get("cards", [])
    title = meta.get("title","Anki-Generator Deck")
    lang  = meta.get("language","en")

    # Stable sort improves reproducibility
    def anchored(c): return 1 if anchored_to_source(c) else 0
    def diff_rank(c): return {"easy":0,"medium":1,"hard":2}.get((c.get("difficulty") or "").lower(), 1)
    cards = sorted(cards, key=lambda c: (-anchored(c), card_kind(c), diff_rank(c), c.get("id","")))

    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    # Choose which cards get audio
    idxs_for_audio = set()
    if tts_policy != "nenhuma" and len(cards) > 0:
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
            if not txt: return idx, ""
            b = synth_audio(txt, lang)
            if not b: return idx, ""
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

        extra_bits = []
        src = c.get("source_ref") or {}
        extra_txt = (c.get("rationale") or "").strip()

        if extra_kind not in ("source","none") and extra_txt:
            lbl = extra_label(extra_kind, lang)
            if lbl: extra_bits.append(f"<div><b>{lbl}</b> {html_escape(extra_txt)}</div>")
        if (src.get("file") or src.get("page_or_time")) and extra_kind in ("source","usage_tip","common_pitfall","mnemonic","self_check"):
            f = html_escape(src.get("file") or "")
            p = html_escape(src.get("page_or_time") or "")
            lbl = extra_label("source", lang)
            if lbl: extra_bits.append(f"<div><b>{lbl}</b> {f} {p}</div>")
        extra = "".join(extra_bits)

        # audio field
        audio_field = audio_map.get(index, "")

        # model & fields
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
def build_sample(payload: Dict[str, Any], sample_n: int = 8) -> Dict[str, Any]:
    _payload = dict(payload)
    _payload["limite_cartoes"] = int(sample_n)
    data = gerar_baralho(_payload)
    cards = dedupe_cards(data.get("cards", []))
    kws = payload.get("domain_keywords", [])
    minima_pct = payload.get("minima_overrides") or goal_mix_minima(payload.get("goal","General learning"))
    filtered, _need = enforce_mix_and_anchoring(
        cards, kws, payload.get("require_anchor", True),
        payload.get("max_qa_pct", 0.5), minima_pct, seed_ids=set()
    )
    # auto-split for preview too (quick signal)
    lang = data.get("deck", {}).get("language", payload.get("idioma_alvo","en"))
    filtered = autosplit_deck(filtered, lang, cap_growth=12)
    data["cards"] = filtered[:sample_n]
    data["deck"]["card_count_planned"] = len(data["cards"])
    return data

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
        f"<div style='border:1px solid #e3e3e3;border-radius:12px;padding:10px;margin-bottom:10px;'>",
        f"<div style='font-weight:600;opacity:.8;margin-bottom:6px;'>{typ}</div>",
        f"<div><b>Q:</b> {front}</div>",
        f"<div style='margin-top:6px;'><b>A:</b> {back}</div>",
    ]
    if hint: parts.append(f"<div style='margin-top:6px;color:#666;'><b>Hint:</b> {hint}</div>")
    if exs:  parts.append(f"<div style='margin-top:6px;'><b>Examples:</b> {exs}</div>")
    if src_str: parts.append(f"<div style='margin-top:6px;color:#444;'><b>Source:</b> {html_escape(src_str)}</div>")
    parts.append("</div>")
    return "".join(parts)

def deck_stats(cards, domain_kws=None):
    kinds = [card_kind(c) for c in cards]
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
      "coverage_pct": cover,
      "anchored_pct": anch
    }

# Editable preview helpers
def cards_to_df(cards):
    if not pd: return None
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

def df_to_cards(df, lang):
    out = []
    for _,r in df.iterrows():
        if not r.get("keep",False): continue
        tags = sanitize_tags([t.strip() for t in str(r.get("tags","")).split(",") if t.strip()])
        cid = f"preseed-{hashlib.md5((str(r['type'])+str(r['front'])).encode()).hexdigest()[:8]}"
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

def csv_to_seed_cards(csv_bytes: bytes, lang: str) -> List[Dict[str,Any]]:
    if not pd: return []
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception:
        return []
    req = {"type","front","back"}
    if not req.issubset(set(df.columns)): return []
    if "keep" not in df.columns: df["keep"] = True
    if "tags" not in df.columns: df["tags"] = ""
    if "hint" not in df.columns: df["hint"] = ""
    return df_to_cards(df, lang)

# -------------------------
# UI — Step-by-step tabs
# -------------------------
st.title("🧠 Anki-Generator")
st.caption("Make high-quality Anki decks with active recall & atomic knowledge. Upload policies or study materials, preview, tweak, and export.")

# Reader mode toggle
reader_mode = st.toggle("Reader mode", value=False, help="Larger fonts & spacing for readability.")
if reader_mode:
    st.markdown("<div class='reader'>", unsafe_allow_html=True)
else:
    st.markdown("<div>", unsafe_allow_html=True)

tabs = st.tabs(["① Topic & Sources", "② Goals & Settings", "③ Preview & Edit", "④ Build & Export"])

# Session state defaults
for key, default in [
    ("sample_data", None),
    ("user_feedback", ""),
    ("sample_n", 8),
    ("approved_cards", []),
    ("seed_random", 0),
    ("materials", []),
    ("urls_loaded", []),
    ("digest", ""),
    ("chosen_sections_meta", []),
]:
    if key not in st.session_state: st.session_state[key] = default

# ---------- Tab 1: Topic & Sources ----------
with tabs[0]:
    st.subheader("Step 1 — Define your topic and add sources")
    st.info("Tip: For organization policies, upload the official PDF(s) and paste any relevant URLs. Enable RAG in Step 2 for focused, anchored cards.")

    col1, col2 = st.columns([2,1])
    with col1:
        topico = st.text_area("Topic / Generation brief",
                              "French past tenses: passé composé, imparfait, plus-que-parfait, passé simple — uses, contrasts, exceptions, mini-scenarios",
                              height=120,
                              help="Describe your deck: scope, outcomes, special focus, or exam/policy targets.")
    with col2:
        deck_title = st.text_input("Deck title",
                                   value=f"{topico[:42]}".strip() or "Anki-Generator Deck",
                                   help="Shown in Anki. Defaults to the topic.")

    url_text = st.text_area("URLs (optional, one per line)", "", height=80,
                            help="Paste public URLs with the core content. We'll extract main text (readability).")
    files = st.file_uploader("Files (PDF/DOCX/TXT/MD) — optional",
                             type=["pdf","docx","txt","md","markdown"], accept_multiple_files=True,
                             help="Attach official policies, study notes, or any reference to anchor cards.")

    colA, colB = st.columns(2)
    with colA:
        load_btn = st.button("Load sources", type="primary", help="Extract text from files and URLs for later use.")
    with colB:
        clear_btn = st.button("Clear sources", help="Forget loaded sources for this session.")

    if clear_btn:
        st.session_state.materials = []
        st.session_state.urls_loaded = []
        st.session_state.digest = ""
        st.session_state.chosen_sections_meta = []
        st.success("Sources cleared.")

    if load_btn:
        mats = ingest_files(files) if files else []
        urls_list = [u.strip() for u in (url_text.splitlines() if url_text else []) if u.strip()]
        if urls_list and not requests:
            st.warning("requests/bs4 not available. Install optional deps to fetch URLs.")
        elif urls_list:
            st.info("Fetching URLs…")
            mats += ingest_urls(urls_list)
        st.session_state.materials = mats
        st.session_state.urls_loaded = urls_list
        st.success(f"Loaded {len(mats)} sources.")

    if st.session_state.materials:
        with st.expander("Preview extracted text (first 500 chars per source)"):
            for m in st.session_state.materials:
                st.markdown(f"**{m['file']}**")
                st.code((m["content"] or "")[:500] + ("..." if len(m["content"])>500 else ""))

# ---------- Tab 2: Goals & Settings ----------
with tabs[1]:
    st.subheader("Step 2 — Choose your goal and tune the generator")
    st.markdown("<span class='small-note'>These settings steer the model to produce the right mix and depth. Hover any control for help.</span>", unsafe_allow_html=True)

    left, right = st.columns([1.2,1])
    with left:
        idioma = st.selectbox("Deck language", ["fr-FR","pt-BR","en-US","es-ES","zh-CN"], index=0,
                              help="Language used on the cards and TTS.")
        nivel  = st.text_input("Level/CEFR (optional)", "B1", help="Add level hints to deck metadata.")
        goal = st.selectbox("Learning goal / focus",
                            ["General learning","Org policy mastery","Exam prep",
                             "Language: Vocabulary","Language: Grammar & Patterns",
                             "Language: Listening-Pronunciation","Language: Reading-CEFR"],
                            index=3,
                            help="Different goals bias card types and content selection.")
        tipos  = st.multiselect("Allowed card types", ["basic","reverse","cloze"],
                                default=["basic","reverse","cloze"],
                                help="We still enforce one-fact-per-card and prefer cloze for patterns.")
        limite = st.slider("Deck size (cards)", 4, 200, 40, 1, help="Target number of cards in the final .apkg.")

        st.session_state.sample_n = st.slider("Preview size", 4, 20, st.session_state.sample_n, 1,
                                              help="How many sample cards to generate for review (fast).")

        # Feedback box
        st.session_state.user_feedback = st.text_area("Refinement instructions (optional)",
            st.session_state.user_feedback,
            height=90,
            help="E.g. 'more cloze on thresholds', 'focus on procedure exceptions', 'B2 grammar patterns, less Q&A, include IPA'.")

    with right:
        # TTS controls
        tts = st.radio("TTS (gTTS)", ["nenhuma","respostas","exemplos","todas"], index=2,
                       help="Which text should be spoken in audio fields.")
        keep_tts_lang_big = st.checkbox("Keep TTS for big decks in language modes", value=True,
                                        help="Avoids auto-disabling TTS when deck > 40 cards, if goal is Language-*.")
        audio_coverage = st.selectbox("TTS coverage", ["Sampled (up to 24 cards)", "All cards (slower)"], index=0,
                                      help="Choose how many cards get audio attached.")
        default_tag = st.text_input("Default tag (optional)", value=slugify(topico) if topico else "",
                                    help="Will be appended to every card for easier filtering in Anki.")

        # Strict / RAG / Mix minima advanced
        with st.expander("Advanced controls"):
            strict = st.checkbox("Strict mode: always reach N cards (top-up rounds)", value=True,
                                 help="If initial generation falls short, we add rounds to hit the target.")
            require_anchor = st.checkbox("Require source anchoring", value=True,
                                         help="Prefer cards that cite uploaded docs/URLs. Helpful for policies.")
            max_qa_pct = st.slider("Max % of Q&A cards", 10, 90, 45, 5,
                                   help="Caps basic Q&A; promotes cloze, scenarios, procedures.")

            st.markdown("**Per-goal minima overrides**")
            use_override = st.checkbox("Override target mix", value=False,
                                       help="Force minimum % for cloze/scenario/procedure.")
            base_minima = goal_mix_minima(goal)
            cloze_min = st.slider("Min Cloze %", 0, 80, int(base_minima["cloze"]*100), 5, disabled=not use_override)
            scenario_min = st.slider("Min Scenario %", 0, 80, int(base_minima["scenario"]*100), 5, disabled=not use_override)
            procedure_min = st.slider("Min Procedure %", 0, 80, int(base_minima["procedure"]*100), 5, disabled=not use_override)

            max_rounds = st.slider("Strict rounds (max)", 1, 8, STRICT_MAX_ROUNDS_DEFAULT, 1,
                                   help="How many top-up attempts to add more cards.")
            batch_size = st.slider("Per-round batch size", 10, 50, STRICT_BATCH_DEFAULT, 5,
                                   help="How many cards to ask per top-up round.")
            seed_val = st.number_input("Random seed (reproducibility)", min_value=0, value=0, step=1,
                                       help="Use the same seed for consistent sampling (TTS & minor variations).")
            if seed_val:
                random.seed(seed_val)

            use_rag = st.checkbox("Use smart context (RAG via TF-IDF)", value=True,
                                   help="Select the most relevant sections from your sources to condition the model.")
            rag_topk = st.slider("RAG sections (top-k)", 3, 12, 6, 1,
                                 help="How many top sections to include. Fewer = tighter context, cheaper requests.")

    # Build digest now (used in preview & build)
    if st.session_state.materials:
        if use_rag:
            st.session_state.digest, st.session_state.chosen_sections_meta = rag_digest(
                st.session_state.materials, topico, st.session_state.user_feedback, top_k=rag_topk)
        else:
            st.session_state.digest = compress_materials_simple(st.session_state.materials)

# ---------- Helpers for later steps ----------
def compute_effective_tts(tts_choice: str, n_cards: int, goal_str: str, keep_big_lang: bool) -> str:
    effective = tts_choice
    is_language_goal = goal_str.lower().startswith("language")
    if tts_choice != "nenhuma" and n_cards > 40 and not (is_language_goal and keep_big_lang):
        effective = "nenhuma"
        st.info("TTS disabled automatically for large decks (>40). Enable 'Keep TTS...' to override for language goals.")
    elif tts_choice != "nenhuma" and n_cards > 40 and (is_language_goal and keep_big_lang):
        st.warning("TTS kept ON for a large language deck. Consider 'Sampled' coverage to keep it fast.")
    return effective

def prepare_payload(effective_tts: str, digest: str, domain_kws: List[str], minima_overrides: Optional[Dict[str,float]]=None, seed_cards=None) -> Dict[str, Any]:
    return {
        "deck_title": deck_title or "Anki-Generator Deck",
        "idioma_alvo": idioma,
        "nivel_proficiencia": nivel,
        "topico": topico.strip(),
        "limite_cartoes": limite,
        "tipos_permitidos": tipos,
        "politica_voz": f"tts={effective_tts}",
        "materiais_digest": digest,
        "extra_policy": extra_kind,
        "goal": goal,
        "max_qa_pct": max_qa_pct/100.0,
        "require_anchor": require_anchor,
        "domain_keywords": domain_kws,
        "user_feedback": st.session_state.user_feedback.strip(),
        "minima_overrides": minima_overrides,
        "seed_cards": seed_cards or []
    }

_kind_map = {
    "Usage tip":"usage_tip", "Common pitfall":"common_pitfall", "Mnemonic":"mnemonic",
    "Self-check":"self_check", "Source":"source", "None":"none",
    "Dica de uso":"usage_tip", "Erro comum":"common_pitfall", "Mnemônico":"mnemonic",
    "Auto-checagem":"self_check", "Fonte":"source", "Nenhum":"none"
}
extra_choice = st.selectbox("Extra section style", ["Dica de uso","Erro comum","Mnemônico","Auto-checagem","Fonte","Nenhum"],
                            index=0, help="Short guidance appended on the answer side.")
extra_kind = _kind_map[extra_choice]

# ---------- Tab 3: Preview & Edit ----------
with tabs[2]:
    st.subheader("Step 3 — Generate a sample and adjust")
    st.markdown("<span class='small-note'>Preview is quick and uses fewer tokens. Approve/edit cards below to seed the final deck.</span>", unsafe_allow_html=True)

    # Domain keywords (for stats later)
    domain_kws = extract_domain_keywords(st.session_state.digest) if st.session_state.digest else extract_domain_keywords(topico)

    col1, col2, col3 = st.columns([1,1,1])
    preview_btn = col1.button("🔎 Generate preview", help="Create a small batch of draft cards for review.")
    regen_btn = col2.button("🔁 Regenerate with my instructions", help="Use 'Refinement instructions' to steer the model and get a new preview.")
    # CSV seed import/export
    seed_upload = col3.file_uploader("Import seed (CSV)", type=["csv"], help="Columns: type,front,back[,hint,tags,keep]")

    if seed_upload is not None and pd:
        seed_from_csv = csv_to_seed_cards(seed_upload.read(), idioma)
        if seed_from_csv:
            st.success(f"CSV seed loaded: {len(seed_from_csv)} cards")
            st.session_state.approved_cards = (st.session_state.get("approved_cards", []) or []) + seed_from_csv

    if preview_btn or regen_btn:
        if not topico.strip():
            st.error("Please fill the Topic.")
        else:
            with st.spinner("Generating preview…"):
                payload = prepare_payload("nenhuma", st.session_state.digest, domain_kws, None, seed_cards=[])
                st.session_state.sample_data = build_sample(payload, sample_n=st.session_state.sample_n)

    if st.session_state.sample_data:
        deck_meta = st.session_state.sample_data.get("deck", {})
        lang = deck_meta.get("language","en")
        cards = st.session_state.sample_data.get("cards", [])[: st.session_state.sample_n]

        if not cards:
            st.warning("No sample cards under current constraints. Try disabling 'Require anchoring' or add more sources.")
        else:
            stats = deck_stats(cards, domain_kws)
            k1,k2,k3,k4,k5,k6 = st.columns(6)
            for k, title, val in [
                (k1,"Cards", f"{stats['total']}"),
                (k2,"Q&A %", f"{stats['qa_pct']:.0f}%"),
                (k3,"Cloze %", f"{stats['cloze_pct']:.0f}%"),
                (k4,"Scenario %", f"{stats['scenario_pct']:.0f}%"),
                (k5,"Anchored %", f"{stats['anchored_pct']:.0f}%"),
                (k6,"Keyword coverage", f"{stats['coverage_pct']:.0f}%"),
            ]:
                with k:
                    st.markdown(f"<div class='kpi'><h3>{title}</h3><div class='val'>{val}</div></div>", unsafe_allow_html=True)

            cols = st.columns(2)
            for i, c in enumerate(cards):
                html = render_card_preview(c, lang)
                (cols[i % 2]).markdown(html, unsafe_allow_html=True)

            if pd:
                st.markdown("**Approve / edit sample cards (kept rows will seed the final deck)**")
                df = cards_to_df(cards)
                edited = st.data_editor(df, use_container_width=True, num_rows="fixed", key="preview_editor")
                st.session_state["approved_cards"] = df_to_cards(edited, lang)

            with st.expander("Download sample CSV"):
                if pd and df is not None:
                    st.download_button("⬇️ Sample CSV", data=df.to_csv(index=False).encode("utf-8"),
                                       file_name="anki_sample.csv", mime="text/csv")

            # Tiny audio audition if desired
            if "nenhuma" not in tts and st.button("▶️ Audition 2 sample audios", help="Hear TTS quality before building the full deck."):
                picks = cards[:2]
                for c in picks:
                    fr, br = orient_q_a(c, lang)
                    txt = choose_tts_text(c, tts, lang, fr, br)
                    if txt:
                        b = synth_audio(txt, lang)
                        if b: st.audio(b, format="audio/mp3")

# ---------- Tab 4: Build & Export ----------
with tabs[3]:
    st.subheader("Step 4 — Build your Anki deck")
    st.markdown("<span class='small-note'>We’ll honor your approved seeds, enforce variety targets, auto-split long answers, and attach TTS if enabled.</span>", unsafe_allow_html=True)

    build_btn = st.button("🏗️ Build .apkg", type="primary", help="Generate the full deck and download the Anki package.")

    if build_btn:
        if not topico.strip():
            st.error("Please fill the Topic.")
        else:
            # Minima overrides
            minima_overrides = None
            if 'use_override' in locals() and use_override:
                minima_overrides = {"cloze":cloze_min/100.0, "scenario":scenario_min/100.0, "procedure":procedure_min/100.0}

            # Combine seeds: approved preview + any CSV
            seed_cards = st.session_state.get("approved_cards", []) or []

            # Domain keywords & TTS decisions
            domain_kws = extract_domain_keywords(st.session_state.digest) if st.session_state.digest else extract_domain_keywords(topico)
            effective_tts = compute_effective_tts(tts, limite, goal, keep_tts_lang_big)
            coverage_mode = "all" if audio_coverage.startswith("All") else "sampled"

            with st.spinner("Building deck…"):
                payload = prepare_payload(effective_tts, st.session_state.digest, domain_kws, minima_overrides, seed_cards=seed_cards)
                prog = st.progress(0.0)
                data = gerar_baralho_estrito(payload, progress=prog, max_rounds=max_rounds, batch_size=batch_size) if strict else gerar_baralho(payload)

                # Light auto-anchoring if missing but sources exist
                if require_anchor and st.session_state.chosen_sections_meta and data.get("cards"):
                    for c in data["cards"]:
                        src = c.get("source_ref") or {}
                        if src.get("file") or src.get("page_or_time"): continue
                        blob = (c.get("front","") + " " + c.get("back","")).lower()
                        best = None; best_score = 0
                        for s in st.session_state.chosen_sections_meta:
                            score = sum(1 for t in set(re.findall(r"\w+", blob)) if t in s["content"].lower())
                            if score > best_score: best = s; best_score = score
                        if best and best_score > 5:
                            c["source_ref"] = {"file": best["file"], "page_or_time": best["title"], "span": None}

                # Final diagnostics
                final_stats = deck_stats(data.get("cards", []), domain_kws)
                k1,k2,k3,k4,k5,k6 = st.columns(6)
                for k, title, val in [
                    (k1,"Cards", f"{final_stats['total']}"),
                    (k2,"Q&A %", f"{final_stats['qa_pct']:.0f}%"),
                    (k3,"Cloze %", f"{final_stats['cloze_pct']:.0f}%"),
                    (k4,"Scenario %", f"{final_stats['scenario_pct']:.0f}%"),
                    (k5,"Anchored %", f"{final_stats['anchored_pct']:.0f}%"),
                    (k6,"Keyword coverage", f"{final_stats['coverage_pct']:.0f}%"),
                ]:
                    with k:
                        st.markdown(f"<div class='kpi'><h3>{title}</h3><div class='val'>{val}</div></div>", unsafe_allow_html=True)

                # Build APKG
                apkg_bytes = build_apkg_bytes(
                    data,
                    tts_policy=effective_tts,
                    extra_kind=extra_kind,
                    tts_coverage=coverage_mode,
                    default_tag=default_tag
                )
                file_name = f"{slugify(deck_title)}_{int(time.time())}.apkg"
                st.success(f"Deck built with {len(data.get('cards', []))} cards. Ready to download.")
                st.download_button("⬇️ Download .apkg", data=apkg_bytes, file_name=file_name, mime="application/octet-stream")

                with st.expander("📦 Export deck JSON / CSV"):
                    st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
                    if pd:
                        rows = []
                        for c in data.get("cards", []):
                            rows.append({
                                "type": c.get("type","basic"),
                                "front": strip_html_to_plain(c.get("front") or c.get("Text","")),
                                "back": strip_html_to_plain(c.get("back") or c.get("BackExtra","")),
                                "hint": c.get("hint",""),
                                "tags": ",".join(sanitize_tags(c.get("tags",[])))
                            })
                        df_final = pd.DataFrame(rows)
                        st.download_button("⬇️ Download CSV", data=df_final.to_csv(index=False).encode("utf-8"),
                                           file_name=f"{slugify(deck_title)}.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)
