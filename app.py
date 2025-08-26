import os, io, json, time, tempfile, hashlib, re
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

import genanki
from gtts import gTTS
from gtts.lang import tts_langs
from pypdf import PdfReader
import docx as docxlib
from unidecode import unidecode

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Anki-Generator", page_icon="🧠", layout="wide")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Defina OPENAI_API_KEY nos Secrets (Streamlit) ou variável de ambiente.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
TEXT_MODEL = "gpt-4o-mini"  # troque para um modelo disponível na sua conta

SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva, integrada ao 'Anki-Generator'.
Princípios: recordação ativa; conhecimento atômico (uma ideia por cartão); frente curta; verso objetivo com 1–2 exemplos.
Use Basic/Reverse/Cloze (evite múltipla escolha). Se houver material do usuário, referencie arquivo/página quando possível.
Saída: JSON:
{
 "deck":{"title":"string","language":"string","level":"string","topic":"string","source_summary":"string","card_count_planned":number},
 "cards":[{"id":"string","type":"basic|reverse|cloze","front":"string","back":"string","hint":"string|null",
           "examples":[{"text":"string","translation":"string|null","notes":"string|null"}],
           "language_fields":{"ipa":"string|null","pinyin":"string|null","morphology":"string|null","register":"string|null"},
           "audio_script":"string|null","tags":["string"],"difficulty":"easy|medium|hard",
           "source_ref":{"file":"string|null","page_or_time":"string|null","span":"string|null"},
           "rationale":"string"}],
 "qa_report":{"duplicates_removed":number,"too_broad_removed":number,"notes":"string"}
}
Regra: 'card_count_planned' deve corresponder ao número real de itens em 'cards'.

Política para o campo extra: siga `payload.extra_policy`.
- Se for "usage_tip", usar `rationale` para uma dica prática (1 frase).
- Se "common_pitfall", `rationale` traz um erro comum e como evitar (1 frase).
- Se "mnemonic", `rationale` com um mnemônico curto (1 frase).
- Se "self_check", `rationale` com mini pergunta de auto-checagem (1 frase).
- Se "source", deixar `rationale` vazio; apenas preencher `source_ref` quando houver.
- Se "none", deixar `rationale` vazio.
"""

# =========================
# UTILS
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
    """
    Remove cercas ```...```, converte \n em <br>. Se cloze=True, preserva {{c1::...}}.
    """
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"^```[\s\S]*?\n|```$", "", s)  # limpa cercas simples
    if cloze:
        s = s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    else:
        s = html_escape(s)
    s = s.replace("\n","<br>")
    return s

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

# gTTS mapping (evita warnings)
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

# genanki tag sanitizer
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

# --- heurística de pergunta + orientação Q/A + rótulos por idioma ---
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

def orient_q_a(card: dict, lang: str) -> tuple[str, str]:
    """Garante que a frente seja uma PERGUNTA, se detectável; senão mantém."""
    f = (card.get("front") or card.get("Text") or "").strip()
    b = (card.get("back")  or card.get("BackExtra") or "").strip()
    fq, bq = looks_like_question(f, lang), looks_like_question(b, lang)
    if fq and not bq:   # ok
        return f, b
    if bq and not fq:   # invertido
        return b, f
    return f, b         # ambíguo → mantém

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
    # default EN
    return {"usage_tip":"Usage tip:", "common_pitfall":"Common pitfall:", "mnemonic":"Mnemonic:", "self_check":"Self-check:", "source":"Source:", "none":""}.get(kind,"")

# =========================
# INGESTÃO DE ARQUIVOS
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

def compress_materials(materials: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    if not materials: return ""
    parts = []; per = max_chars // max(1, len(materials))
    for m in materials:
        chunk = (m["content"] or "")[:per]
        parts.append(f"# {m['file']}\n{chunk}")
    return "\n\n".join(parts)[:max_chars]

# =========================
# OPENAI → GERAÇÃO (básico)
# =========================
def gerar_baralho(payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":json.dumps(payload, ensure_ascii=False)}
        ],
        temperature=0,
        response_format={"type":"json_object"}
    )
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}

    cards = data.get("cards", [])
    if not isinstance(cards, list): cards = []

    # Normaliza cloze e conteúdo HTML
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
    data["deck"].setdefault("title", "Anki-Generator Deck")
    data["deck"].setdefault("language", payload.get("idioma_alvo","en"))
    data["deck"].setdefault("level", payload.get("nivel_proficiencia",""))
    data["deck"].setdefault("topic", payload.get("topico",""))
    data["deck"]["card_count_planned"] = len(cards)
    data["cards"] = cards
    return data

# =========================
# DEDUPE + TOP-UP (modo estrito)
# =========================
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
        seen.add(sig); out.append(c)
    return out

def gerar_cartoes_adicionais(payload: dict, ja_gerados: list, faltantes: int, lote: int = 20) -> list:
    novos_total = []; restantes = max(0, int(faltantes))
    resumo = []
    for c in ja_gerados or []:
        resumo.append({"type": c.get("type"),
                       "front": c.get("front") or c.get("Text"),
                       "back":  c.get("back")  or c.get("BackExtra"),
                       "tags":  c.get("tags", [])})
    while restantes > 0:
        pedir = min(restantes, lote)
        sys_addendum = (
            SYSTEM_PROMPT +
            "\n\nINSTRUÇÃO ADICIONAL: Gere EXATAMENTE o número solicitado e responda SOMENTE com JSON no formato "
            '{"cards":[{...}]}' " (sem 'deck' e sem 'qa_report'). Não repita nenhum cartão existente."
        )
        pedido = {
            "pedido": f"Gerar exatamente {pedir} novos cartões NÃO-DUPLICADOS, mantendo active recall e conhecimento atômico.",
            "payload_base": {
                "idioma_alvo": payload.get("idioma_alvo"),
                "nivel_proficiencia": payload.get("nivel_proficiencia"),
                "topico": payload.get("topico"),
                "tipos_permitidos": payload.get("tipos_permitidos"),
                "politica_voz": payload.get("politica_voz"),
                "materiais_digest": payload.get("materiais_digest",""),
                "extra_policy": payload.get("extra_policy","usage_tip")
            },
            "cartoes_ja_gerados_resumo": resumo
        }
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role":"system","content":sys_addendum},
                      {"role":"user","content":json.dumps(pedido, ensure_ascii=False)}],
            temperature=0,
            response_format={"type":"json_object"}
        )
        try:
            data = json.loads(resp.choices[0].message.content)
        except Exception:
            data = {}
        novos = data.get("cards", [])
        if not isinstance(novos, list): novos = []

        for c in novos:
            if not isinstance(c, dict): continue
            typ = (c.get("type") or "basic").lower()
            if typ == "cloze":
                if not c.get("front") and c.get("Text"): c["front"] = c["Text"]
                if not c.get("back")  and c.get("BackExtra"): c["back"] = c["BackExtra"]
                c["front"] = normalize_text_for_html(c.get("front",""), cloze=True)
                c["back"]  = normalize_text_for_html(c.get("back",""))
            else:
                c["front"] = normalize_text_for_html(c.get("front",""))
                c["back"]  = normalize_text_for_html(c.get("back",""))

        candidatos = dedupe_cards((ja_gerados or []) + novos)
        have = {_card_signature(x) for x in (ja_gerados or [])}
        realmente_novos = [c for c in candidatos if _card_signature(c) not in have]
        ja_gerados = dedupe_cards((ja_gerados or []) + realmente_novos)
        novos_total += realmente_novos
        restantes = max(0, faltantes - len(novos_total))
        if pedir > 0 and len(realmente_novos) == 0:
            break
    return novos_total[:faltantes]

def gerar_baralho_estrito(payload: dict, max_rodadas: int = 4, lote: int = 20) -> dict:
    desired = int(payload.get("limite_cartoes", 20))
    base = gerar_baralho(payload)
    cards = dedupe_cards(base.get("cards", []))
    rod = 0
    while len(cards) < desired and rod < max_rodadas:
        faltam = desired - len(cards)
        novos = gerar_cartoes_adicionais(payload, cards, faltam, lote=lote)
        cards = dedupe_cards(cards + novos)
        rod += 1
    base["cards"] = cards[:desired]
    base.setdefault("deck", {})
    base["deck"]["card_count_planned"] = len(base["cards"])
    return base

# =========================
# ANKI (genanki) — MODELOS v2 (IDs novos)
# =========================
def stable_model_id(name: str, version: int = 2) -> int:
    h = hashlib.sha1(f"{name}-v{version}".encode("utf-8")).hexdigest()
    return int(h[:10], 16)

COMMON_CSS = """
.card { font-family: -apple-system, Segoe UI, Roboto, Arial; font-size: 20px; text-align: left; color: #222; background: #fff; }
.front { font-size: 1.05em; line-height: 1.45; }
.back  { line-height: 1.5; }
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
    stable_model_id("Anki-Generator Basic", version=2), "Anki-Generator Basic (v2)",
    fields=[{"name":"Front"},{"name":"Back"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Card 1",
        "qfmt":"<div class='front'>{{Front}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"{{FrontSide}}<hr id='answer'><div class='back'>{{Back}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div class='audio'>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
    }],
    css=COMMON_CSS
)

MODEL_REVERSE = genanki.Model(
    stable_model_id("Anki-Generator Reverse", version=2), "Anki-Generator Reverse (v2)",
    fields=[{"name":"Front"},{"name":"Back"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Reverse Only",
        "qfmt":"<div class='front'>{{Back}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"<div class='front'>{{Back}}</div><hr id='answer'><div class='back'>{{Front}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div class='audio'>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
    }],
    css=COMMON_CSS
)

MODEL_CLOZE = genanki.Model(
    stable_model_id("Anki-Generator Cloze", version=2), "Anki-Generator Cloze (v2)",
    fields=[{"name":"Text"},{"name":"BackExtra"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Cloze Card",
        "qfmt":"<div class='front'>{{cloze:Text}}</div>{{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"<div class='front'>{{cloze:Text}}</div><hr id='answer'><div class='back'>{{BackExtra}}</div>"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div class='audio'>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
    }],
    css=COMMON_CSS,
    model_type=genanki.Model.CLOZE
)

def build_deck_id(title: str) -> int:
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return int(h[:10], 16)

def build_apkg_bytes(deck_json: Dict[str, Any], tts_policy: str = "exemplos", extra_kind: str = "usage_tip") -> bytes:
    meta = deck_json.get("deck", {})
    cards = deck_json.get("cards", [])
    title = meta.get("title","Anki-Generator Deck")
    lang  = meta.get("language","en")

    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    def add_note(c: Dict[str, Any]):
        ctype = (c.get("type") or "basic").lower()
        # Trate reverse como basic para manter pergunta→resposta
        if ctype == "reverse":
            ctype = "basic"

        hint = html_escape(c.get("hint") or "")
        examples_html = examples_to_html(c.get("examples"))

        # --- seção EXTRA configurável ---
        extra_bits = []
        src = c.get("source_ref") or {}
        extra_txt = (c.get("rationale") or "").strip()

        if extra_kind != "source" and extra_kind != "none" and extra_txt:
            lbl = extra_label(extra_kind, lang)
            if lbl:
                extra_bits.append(f"<div><b>{lbl}</b> {html_escape(extra_txt)}</div>")

        if (src.get("file") or src.get("page_or_time")) and extra_kind in ("source","usage_tip","common_pitfall","mnemonic","self_check"):
            f = html_escape(src.get("file") or "")
            p = html_escape(src.get("page_or_time") or "")
            lbl = extra_label("source", lang)
            if lbl:
                extra_bits.append(f"<div><b>{lbl}</b> {f} {p}</div>")

        extra = "".join(extra_bits)

        audio_field = ""
        wants_tts = tts_policy != "nenhuma"
        if wants_tts and c.get("audio_script"):
            bts = synth_audio(c["audio_script"], lang)
            if bts:
                mp3_path = os.path.join(tmpdir, f"tts_{int(time.time()*1000)}.mp3")
                with open(mp3_path, "wb") as f: f.write(bts)
                media_files.append(mp3_path)
                audio_field = f"[sound:{os.path.basename(mp3_path)}]"

        if ctype == "cloze":
            text = c.get("front","")  # já normalizado
            back_extra = c.get("back","") or ""
            note = genanki.Note(
                model=MODEL_CLOZE,
                fields=[text, back_extra, hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
        else:
            # Orienta para que a frente seja a PERGUNTA
            front_raw, back_raw = orient_q_a(c, lang)
            front = normalize_text_for_html(front_raw)
            back  = normalize_text_for_html(back_raw)
            note = genanki.Note(
                model=MODEL_BASIC,
                fields=[front, back, hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
        deck.add_note(note)

    for c in cards:
        if isinstance(c, dict): add_note(c)

    pkg = genanki.Package(deck)
    if media_files: pkg.media_files = media_files

    with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as tmp:
        pkg.write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            apkg_bytes = f.read()
    return apkg_bytes

# =========================
# UI
# =========================
st.title("🧠 Anki-Generator")
st.caption("Gere baralhos de Anki por tema ou a partir de documentos. (Cartões seguem recordação ativa + conhecimento atômico.)")

col1, col2 = st.columns([1,2])
with col1:
    idioma = st.selectbox("Idioma", ["fr-FR","pt-BR","en-US","es-ES","zh-CN"], index=0)
    nivel  = st.text_input("Nível", "B1")
    limite = st.slider("Qtde de cartões", 4, 120, 20, 1)
    tipos  = st.multiselect("Tipos de cartão", ["basic","reverse","cloze"], default=["basic","reverse","cloze"])
    tts    = st.radio("TTS (gTTS)", ["nenhuma","respostas","exemplos","todas"], index=2)
    strict = st.checkbox("Exigir exatamente N (completar se vierem menos)", value=True)
    extra_choice = st.selectbox(
        "Seção extra do cartão",
        ["Dica de uso","Erro comum","Mnemônico","Auto-checagem","Fonte","Nenhum"],
        index=0
    )
    _kind_map = {
        "Dica de uso":"usage_tip", "Erro comum":"common_pitfall", "Mnemônico":"mnemonic",
        "Auto-checagem":"self_check", "Fonte":"source", "Nenhum":"none"
    }
    extra_kind = _kind_map[extra_choice]
with col2:
    topico = st.text_area("Tópico", "Tempos do passado em francês: passé composé, imparfait, plus-que-parfait, passé simple – usos, diferenças e exemplos", height=140)
    files  = st.file_uploader("Arquivos (PDF/DOCX/TXT/MD – opcional)", type=["pdf","docx","txt","md","markdown"], accept_multiple_files=True)

btn = st.button("Gerar baralho (.apkg)", type="primary", use_container_width=True)

if btn:
    if not topico.strip():
        st.error("Preencha o campo Tópico.")
        st.stop()

    with st.spinner("Gerando cartões…"):
        materials = ingest_files(files) if files else []
        digest = compress_materials(materials) if materials else ""
        payload = {
            "idioma_alvo": idioma,
            "nivel_proficiencia": nivel,
            "topico": topico.strip(),
            "limite_cartoes": limite,
            "tipos_permitidos": tipos,
            "politica_voz": f"tts={tts}",
            "materiais_digest": digest,
            "extra_policy": extra_kind
        }
        data = gerar_baralho_estrito(payload) if strict else gerar_baralho(payload)
        apkg_bytes = build_apkg_bytes(data, tts_policy=tts, extra_kind=extra_kind)
        file_name = f"Anki-Generator_{int(time.time())}.apkg"

        st.success(f"Baralho gerado com {len(data.get('cards', []))} cartões!")
        st.download_button("⬇️ Baixar .apkg", data=apkg_bytes, file_name=file_name, mime="application/octet-stream")
        with st.expander("🔍 Ver JSON gerado (debug)"):
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
