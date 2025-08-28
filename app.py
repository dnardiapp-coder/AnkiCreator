import os, io, json, time, tempfile, hashlib, re, random, textwrap
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from openai import OpenAI

import genanki
from gtts import gTTS
from gtts.lang import tts_langs
from pypdf import PdfReader
import docx as docxlib
from unidecode import unidecode

# Optional deps (best-effort)
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

# =========================
# CONFIG & GLOBALS
# =========================
st.set_page_config(page_title="Anki-Generator", page_icon="🧠", layout="wide")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Defina OPENAI_API_KEY nos Secrets (Streamlit) ou variável de ambiente.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

TEXT_MODEL = "gpt-4o-mini"
STRICT_MAX_ROUNDS_DEFAULT = 4
STRICT_BATCH_DEFAULT = 20
STRICT_HARD_TIMEOUT = 180
MAX_AUDIO_FILES = 24
AUDIO_CHAR_LIMIT = 400

# =========================
# PROMPT (with feedback + Flashcard Guide)
# =========================
SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva, integrada ao 'Anki-Generator'.

OBJETIVO
- Gerar cartões úteis, específicos e de alta qualidade, ancorados no tópico e nos materiais do usuário (quando houver).
- Práticas baseadas em evidência: recordação ativa, conhecimento atômico, exemplos concretos, interleaving/variação, contraste/erro comum, foco em transferência para uso real.

CONSIDERE feedback do usuário (payload.user_feedback), quando presente, para ajustar foco, granularidade, estilo, exemplos, tipos de cartão, âncoras e cobertura.

FINS (payload.goal)
- "General learning": equilíbrio entre definição, cloze, cenários e procedimentos.
- "Org policy mastery": foque itens práticos: seções/IDs, prazos, thresholds/valores, exceções, responsáveis, aprovações, sanções, conformidade, auditoria, SLA/OLA. Muitos cenários e procedimentos.
- "Exam prep": foco em pontos cobrados/pegadinhas/variações; >=30% cloze, >=20% cenários; inclua erros comuns.
- "Language: Vocabulary": collocations, regência/partículas, falsos cognatos; cloze de palavras, exemplos bilíngues, frase de uso; audio_script apropriado.
- "Language: Grammar & Patterns": contraste de padrões, cloze morfossintático, condições/restrições; exemplos mínimos; audio_script com frases-alvo.
- "Language: Listening-Pronunciation": priorize áudio, IPA quando aplicável; cloze auditiva (palavra/chunk); script natural.
- "Language: Reading-CEFR": cloze de conectores, tempos, referência pronominal; microtrechos com inferência.

MIX & VARIEDADE
- Tipos: basic | reverse | cloze (evite múltipla escolha).
- Varie conteúdos (inclusive em basic):
  • Definição/critério diagnóstico com exemplo.
  • Cloze (omita termo, número, data, cláusula, passo).
  • Cenário/mini-caso → decisão/aplicação da regra/política.
  • Procedimento/checklist (ordem, responsáveis, prazos).
  • Contraste/exceções/thresholds (se/então, valores, datas).
- Respeite payload.max_qa_pct: no máximo essa fração de Q&A diretos; favoreça cloze/cenário/procedimento conforme meta.

ESPECIFICIDADE & ANCORAGEM
- Use payload.materiais_digest quando existir. Ao derivar do material, preencha source_ref.file e page_or_time.
- Utilize payload.domain_keywords nos enunciados/respostas para termos do domínio; evite generalidades.
- Racional (rationale) segue payload.extra_policy (dica, erro comum, mnemônico, auto-checagem, fonte, nenhum).

🧠 GUIA DE CRIAÇÃO DE FLASHCARDS (INTEGRADO)
Parte 1 – Princípios (As 20 Regras)
1) Uma informação por cartão. 2) Perguntas específicas. 3) Linguagem clara/simples.
4) Resposta única e inequívoca. 5) Cartões concisos. 6) Duas vias quando fizer sentido.
7) Promova recordação ativa (Q/cloze > reconhecimento). 8) Resposta rápida (2–5s).
9) Foco em conhecimento estável. 10) Conceitos precisos; evite vagos.
11) Reescreva com suas palavras (clareza). 12) Exemplos p/ ideias abstratas.
13) Pistas mínimas somente quando necessárias (hint curto). 14) Estruture em camadas (básico→complexo).
15) Quebre assuntos grandes em subcartões. 16) Não confunda reconhecimento com recall.
17) Cloze com sabedoria (um alvo por cloze). 18) Refine cartões ruins (marque no qa_report).
19) Personalize aos objetivos do usuário (payload.goal). 20) Teste para uso real (contextos autênticos).

Parte 2 – Técnicas
- Basic Q&A: fatos simples e curtos.
- Cloze deletion: palavra-chave/termo/número/cláusula em contexto.
- Reverse cards: opcional quando reforça bidirecionalidade.
- Step-by-step process: um passo por cartão (ordem/quem faz/prazo).
- Component/part: listas → itens individuais.
- Cause-effect: “Se X, então Y” (condições/efeitos).
- Comparison/contrast: diferenças essenciais (diagnóstico).

Parte 3 – Erros a evitar
- Cartões carregados; perguntas vagas; múltiplas respostas; cópia literal sem compreensão;
- Múltipla escolha em excesso; não editar cartões ruins; contexto demais; redação confusa.

Parte 4 – Modelos de boa pergunta
- “O que é X?”; “X tem quantos Y?”; “Se X, então Y?”; “Por que X?”;
- “Quais componentes de X?”; “O que acontece em X?”.

SAÍDA (OBRIGATÓRIO)
- JSON:
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

# =========================
# UTILITÁRIOS
# =========================
def slugify(text: str, maxlen: int = 64) -> str:
    t = unidecode(text or "").lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t[:maxlen] or f"slug-{int(time.time())}"

def html_escape(s: Optional[str]) -> str:
    if not s: return ""
    return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"))

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

# gTTS helpers
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

# tags
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

# Q/A heuristic
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

# =========================
# INGESTÃO DE ARQUIVOS & URLS
# =========================
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
    """Split by markdown headers or large paragraphs."""
    sections = []
    # split by headers
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
    """Return a condensed digest built from top-k relevant sections + sections meta."""
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
    parts = []; per = max_chars // max(1, len(materials))
    for m in materials:
        chunk = (m["content"] or "")[:per]
        parts.append(f"# {m['file']}\n{chunk}")
    return "\n\n".join(parts)[:max_chars]

# =========================
# MIX MINIMA (goal & overrides)
# =========================
def goal_mix_minima(goal: str) -> Dict[str, float]:
    g = (goal or "").lower()
    if "policy" in g: return {"cloze": 0.25, "scenario": 0.35, "procedure": 0.25}
    if "exam prep" in g: return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}
    if "vocabulary" in g: return {"cloze": 0.35, "scenario": 0.05, "procedure": 0.00}
    if "grammar" in g or "patterns" in g: return {"cloze": 0.50, "scenario": 0.10, "procedure": 0.00}
    if "listening" in g or "pronunciation" in g: return {"cloze": 0.30, "scenario": 0.10, "procedure": 0.00}
    if "reading" in g or "cefr" in g: return {"cloze": 0.40, "scenario": 0.10, "procedure": 0.00}
    return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}

# =========================
# OPENAI WRAPPER + JSON SAFE
# =========================
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
    if len(strip_html_to_plain(c.get("back",""))) > 360:  # fast-answer heuristic
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

# =========================
# VARIEDADE/ÂNCORA & DEDUPE
# =========================
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
        # near-dup check vs small sample of existing
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
        # Remove shortest-front QAs first (non-seed)
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

# =========================
# TOP-UP (STRICT MODE)
# =========================
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
        "Use payload_base.domain_keywords nos enunciados/respostas. "
        "Se require_anchor=true e houver materiais, preencha source_ref.file e page_or_time."
    )

    pedido = {
        "pedido": f"Gerar exatamente {pedir} cartões NÃO-DUPLICADOS, variados (cloze/cenário/procedimento/definição-criterial), mantendo active recall e atômicos.",
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

# =========================
# ANKI MODELS (v3)
# =========================
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

# ---------- TTS selection ----------
def choose_tts_text(card: Dict[str, Any], policy: str, lang: str, front_raw: str, back_raw: str) -> Optional[str]:
    if policy == "nenhuma": return None
    audio_script = (card.get("audio_script") or "").strip()
    ex = ""
    if isinstance(card.get("examples"), list) and card["examples"]:
        ex = (card["examples"][0].get("text") or "").strip()
    p = policy.lower()
    if p == "todas":
        candidates = [audio_script, ex, back_raw, front_raw]
    elif p == "respostas":
        candidates = [audio_script, back_raw, ex, front_raw]
    elif p == "exemplos":
        candidates = [ex, audio_script, back_raw, front_raw]
    else:
        candidates = [audio_script, back_raw, ex, front_raw]
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

    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    idxs_for_audio = set()
    if tts_policy != "nenhuma" and len(cards) > 0:
        if tts_coverage.lower().startswith("all"):
            idxs_for_audio = set(range(len(cards)))
        else:
            idxs_for_audio = set(random.sample(range(len(cards)), min(MAX_AUDIO_FILES, len(cards))))

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

        # audio
        audio_field = ""
        front_raw, back_raw = orient_q_a(c, lang)
        if index in idxs_for_audio:
            tts_text = choose_tts_text(c, tts_policy, lang, front_raw, back_raw)
            if tts_text:
                bts = synth_audio(tts_text, lang)
                if bts:
                    mp3_path = os.path.join(tmpdir, f"tts_{int(time.time()*1000)}_{index}.mp3")
                    with open(mp3_path, "wb") as f: f.write(bts)
                    media_files.append(mp3_path)
                    audio_field = f"[sound:{os.path.basename(mp3_path)}]"

        # model selection
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

# =========================
# PREVIEW & STATS
# =========================
def build_sample(payload: Dict[str, Any], sample_n: int = 8) -> Dict[str, Any]:
    _payload = dict(payload)
    _payload["limite_cartoes"] = int(sample_n)
    data = gerar_baralho(_payload)
    cards = dedupe_cards(data.get("cards", []))
    kws = payload.get("domain_keywords", [])
    minima_pct = payload.get("minima_overrides") or goal_mix_minima(payload.get("goal","General learning"))
    seed_ids = set()  # no seeds in quick preview
    filtered, _need = enforce_mix_and_anchoring(
        cards, kws, payload.get("require_anchor", True),
        payload.get("max_qa_pct", 0.5), minima_pct, seed_ids
    )
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
        f"<div style='border:1px solid #e3e3e3;border-radius:10px;padding:10px;margin-bottom:10px;'>",
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

def deck_stats(cards):
    kinds = [card_kind(c) for c in cards]
    total = len(cards) or 1
    return {
      "total": total,
      "qa_pct": 100*sum(1 for k in kinds if k=="qa")/total,
      "cloze_pct": 100*sum(1 for k in kinds if k=="cloze")/total,
      "scenario_pct": 100*sum(1 for k in kinds if k=="scenario")/total,
      "procedure_pct": 100*sum(1 for k in kinds if k=="procedure")/total,
      "avg_back_len": sum(len(strip_html_to_plain(c.get("back",""))) for c in cards)/total
    }

# Editable preview helpers (CSV/JSON I/O)
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
    # add keep default true if missing
    if "keep" not in df.columns: df["keep"] = True
    if "tags" not in df.columns: df["tags"] = ""
    if "hint" not in df.columns: df["hint"] = ""
    return df_to_cards(df, lang)

# =========================
# UI
# =========================
st.title("🧠 Anki-Generator — Pro")
st.caption("Active recall + conhecimento atômico • RAG opcional • Pré-visualização editável • Export/Import CSV • TTS controlado")

# Session state
for key, default in [
    ("sample_data", None),
    ("user_feedback", ""),
    ("sample_n", 8),
    ("approved_cards", []),
    ("seed_random", 0),
]:
    if key not in st.session_state: st.session_state[key] = default

# Basic controls
left, right = st.columns([1,2])
with left:
    deck_title = st.text_input("Deck title", "Anki-Generator Deck")
    default_tag = st.text_input("Default tag (optional)", "")
    idioma = st.selectbox("Idioma (deck)", ["fr-FR","pt-BR","en-US","es-ES","zh-CN"], index=0)
    nivel  = st.text_input("Nível/CEFR (opcional)", "B1")
    limite = st.slider("Qtde de cartões (deck final)", 4, 200, 40, 1)
    tipos  = st.multiselect("Tipos de cartão", ["basic","reverse","cloze"], default=["basic","reverse","cloze"])
    tts    = st.radio("TTS (gTTS)", ["nenhuma","respostas","exemplos","todas"], index=2)

    goal = st.selectbox(
        "Goal / Focus",
        [
            "General learning",
            "Org policy mastery",
            "Exam prep",
            "Language: Vocabulary",
            "Language: Grammar & Patterns",
            "Language: Listening-Pronunciation",
            "Language: Reading-CEFR",
        ],
        index=3
    )

    seed_val = st.number_input("Random seed (optional)", min_value=0, value=0, step=1)
    if seed_val:
        random.seed(seed_val)
        st.session_state["seed_random"] = seed_val

with right:
    topico = st.text_area(
        "Tópico / Diretrizes de geração",
        "Tempos do passado em francês: passé composé, imparfait, plus-que-parfait, passé simple – usos, diferenças e exemplos",
        height=120
    )
    url_text = st.text_area("URLs (opcional, um por linha)", "", height=80)
    files  = st.file_uploader(
        "Arquivos (PDF/DOCX/TXT/MD – opcional; para políticas, suba os documentos oficiais)",
        type=["pdf","docx","txt","md","markdown"], accept_multiple_files=True
    )

# Advanced options
with st.expander("⚙️ Advanced"):
    strict = st.checkbox("Exigir exatamente N (Strict mode com top-up)", value=True)
    require_anchor = st.checkbox("Require source anchoring (from uploaded docs/URLs)", value=True)
    max_qa_pct = st.slider("Max % of Q&A cards", 10, 90, 45, 5)

    st.markdown("**Per-goal minima overrides (% of deck)**")
    use_override = st.checkbox("Override goal minima", value=False)
    base_minima = goal_mix_minima(goal)
    cloze_min = st.slider("Min Cloze %", 0, 80, int(base_minima["cloze"]*100), 5, disabled=not use_override)
    scenario_min = st.slider("Min Scenario %", 0, 80, int(base_minima["scenario"]*100), 5, disabled=not use_override)
    procedure_min = st.slider("Min Procedure %", 0, 80, int(base_minima["procedure"]*100), 5, disabled=not use_override)

    st.markdown("**Strict parameters**")
    max_rounds = st.slider("Max strict rounds", 1, 8, STRICT_MAX_ROUNDS_DEFAULT, 1)
    batch_size = st.slider("Per-round batch size", 10, 50, STRICT_BATCH_DEFAULT, 5)

    st.markdown("**TTS preferences (large decks)**")
    keep_tts_lang_big = st.checkbox("Keep TTS for big decks in language modes", value=True)
    audio_coverage = st.selectbox("TTS coverage", ["Sampled (up to 24 cards)", "All cards (slower)"], index=0)

    st.markdown("**RAG context**")
    use_rag = st.checkbox("Use smart context (RAG) with TF-IDF selection", value=True)
    rag_topk = st.slider("RAG sections (top-k)", 3, 12, 6, 1)

# Preview controls
st.session_state.sample_n = st.slider("Preview: number of sample cards", 4, 20, st.session_state.sample_n, 1)
with st.expander("Optional: refinement instructions for the model (used for sample and final)"):
    st.session_state.user_feedback = st.text_area(
        "Refine style/content (e.g., “more cloze on thresholds”, “focus on procedure exceptions”, “B2 grammar patterns, less Q&A, include IPA”)",
        st.session_state.user_feedback,
        height=110
    )

# CSV seed import/export
seed_upload = st.file_uploader("Seed cards from CSV (columns: type,front,back[,hint,tags,keep])", type=["csv"])

btn_cols = st.columns(3)
preview_btn = btn_cols[0].button("🔎 Generate sample preview")
regenerate_btn = btn_cols[1].button("🔁 Regenerate sample with feedback")
build_btn = btn_cols[2].button("🏗️ Build full deck (.apkg)", type="primary")

# ------- PROCESSAMENTO --------
def compute_effective_tts(tts_choice: str, n_cards: int, goal_str: str, keep_big_lang: bool) -> str:
    effective = tts_choice
    is_language_goal = goal_str.lower().startswith("language")
    if tts_choice != "nenhuma" and n_cards > 40 and not (is_language_goal and keep_big_lang):
        effective = "nenhuma"
        st.info("TTS disabled automatically for large decks (>40). Enable the checkbox to keep TTS in language modes.")
    elif tts_choice != "nenhuma" and n_cards > 40 and (is_language_goal and keep_big_lang):
        st.warning("TTS kept ON for a large language deck. Consider 'Sampled' coverage to keep it fast.")
    return effective

def prepare_payload(effective_tts: str, digest: str, domain_kws: List[str], minima_overrides: Optional[Dict[str,float]]=None, seed_cards=None) -> Dict[str, Any]:
    return {
        "deck_title": deck_title,
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

# map extra selection
_kind_map = {
    "Dica de uso":"usage_tip", "Erro comum":"common_pitfall", "Mnemônico":"mnemonic",
    "Auto-checagem":"self_check", "Fonte":"source", "Nenhum":"none"
}
extra_choice = st.selectbox("Seção extra do cartão", list(_kind_map.keys()), index=0)
extra_kind = _kind_map[extra_choice]

# Material ingest
materials = ingest_files(files) if files else []
urls = [u for u in (url_text.splitlines() if url_text else []) if u.strip()]
if urls:
    if not requests:
        st.warning("Requests/BS4 not available. Install optional dependencies to use URLs.")
    else:
        with st.spinner("Fetching URLs…"):
            materials += ingest_urls(urls)

# RAG or simple digest
digest = ""
chosen_sections_meta = []
if materials:
    if use_rag:
        digest, chosen_sections_meta = rag_digest(materials, topico, st.session_state.user_feedback, top_k=rag_topk)
    else:
        digest = compress_materials_simple(materials)
# domain keywords
domain_kws = extract_domain_keywords(digest) if digest else extract_domain_keywords(topico)

# TTS effective
effective_tts = compute_effective_tts(tts, limite, goal, keep_tts_lang_big)
coverage_mode = "all" if audio_coverage.startswith("All") else "sampled"

# Seed from CSV
seed_from_csv = []
if seed_upload is not None:
    if not pd:
        st.error("Pandas não disponível para ler CSV.")
    else:
        seed_from_csv = csv_to_seed_cards(seed_upload.read(), idioma)
        if seed_from_csv:
            st.success(f"CSV seed loaded: {len(seed_from_csv)} cards")

# Preview generation
if preview_btn or regenerate_btn:
    if not topico.strip():
        st.error("Preencha o campo Tópico.")
        st.stop()
    with st.spinner("Generating sample…"):
        payload = prepare_payload("nenhuma", digest, domain_kws, None, seed_cards=[])
        st.session_state.sample_data = build_sample(payload, sample_n=st.session_state.sample_n)

# Show preview
if st.session_state.sample_data:
    st.subheader("Preview")
    deck_meta = st.session_state.sample_data.get("deck", {})
    lang = deck_meta.get("language","en")
    cards = st.session_state.sample_data.get("cards", [])[: st.session_state.sample_n]

    if not cards:
        st.warning("No sample cards generated under current constraints. Try unchecking ‘Require anchoring’, increase preview size, or add more materials.")
    else:
        stats = deck_stats(cards)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Cards", stats["total"])
        c2.metric("Q&A %", f"{stats['qa_pct']:.0f}%")
        c3.metric("Cloze %", f"{stats['cloze_pct']:.0f}%")
        c4.metric("Scenario %", f"{stats['scenario_pct']:.0f}%")
        c5.metric("Avg answer chars", f"{stats['avg_back_len']:.0f}")
        if stats["qa_pct"] > max_qa_pct:
            st.warning("Q&A exceeds the configured maximum. Add feedback or lower max %.")
        if stats["avg_back_len"] > 360:
            st.info("Answers look long. Ask for more cloze/procedure cards in the feedback.")

        cols = st.columns(2)
        for i, c in enumerate(cards):
            html = render_card_preview(c, lang)
            (cols[i % 2]).markdown(html, unsafe_allow_html=True)

        # Editable table + seed approval
        if pd:
            df = cards_to_df(cards)
            edited = st.data_editor(df, use_container_width=True, num_rows="fixed", key="preview_editor")
            st.session_state["approved_cards"] = df_to_cards(edited, lang)

        exp = st.expander("🔍 Sample JSON / Export CSV")
        with exp:
            st.code(json.dumps(st.session_state.sample_data, ensure_ascii=False, indent=2), language="json")
            # CSV export
            if pd:
                csv_df = cards_to_df(cards)
                if csv_df is not None:
                    st.download_button("⬇️ Download sample CSV", data=csv_df.to_csv(index=False).encode("utf-8"),
                                       file_name="anki_sample.csv", mime="text/csv")

# Build full deck
if build_btn:
    if not topico.strip():
        st.error("Preencha o campo Tópico.")
        st.stop()

    # minima overrides
    minima_overrides = None
    if use_override:
        minima_overrides = {"cloze":cloze_min/100.0, "scenario":scenario_min/100.0, "procedure":procedure_min/100.0}

    # combine seeds: approved preview + CSV
    seed_cards = (st.session_state.get("approved_cards", []) or []) + (seed_from_csv or [])

    with st.spinner("Building full deck…"):
        payload = prepare_payload(effective_tts, digest, domain_kws, minima_overrides, seed_cards=seed_cards)
        prog = st.progress(0.0)
        data = gerar_baralho_estrito(payload, progress=prog, max_rounds=max_rounds, batch_size=batch_size) if strict else gerar_baralho(payload)

        # If anchoring was required but missing, try light auto-anchoring by overlap against chosen sections
        if require_anchor and chosen_sections_meta and data.get("cards"):
            for c in data["cards"]:
                src = c.get("source_ref") or {}
                if src.get("file") or src.get("page_or_time"): continue
                blob = (c.get("front","") + " " + c.get("back","")).lower()
                best = None; best_score = 0
                for s in chosen_sections_meta:
                    score = sum(1 for t in set(re.findall(r"\w+", blob)) if t in s["content"].lower())
                    if score > best_score:
                        best = s; best_score = score
                if best and best_score > 5:
                    c["source_ref"] = {"file": best["file"], "page_or_time": best["title"], "span": None}

        apkg_bytes = build_apkg_bytes(
            data,
            tts_policy=effective_tts,
            extra_kind=extra_kind,
            tts_coverage=coverage_mode,
            default_tag=default_tag
        )
        file_name = f"{slugify(deck_title)}_{int(time.time())}.apkg"

        st.success(f"Baralho gerado com {len(data.get('cards', []))} cartões!")
        st.download_button("⬇️ Baixar .apkg", data=apkg_bytes, file_name=file_name, mime="application/octet-stream")

        # Export final JSON & CSV
        with st.expander("📦 Export deck JSON / CSV"):
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
            if pd:
                # Convert to CSV for export
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
                st.download_button("⬇️ Download final CSV", data=df_final.to_csv(index=False).encode("utf-8"),
                                   file_name=f"{slugify(deck_title)}.csv", mime="text/csv")
