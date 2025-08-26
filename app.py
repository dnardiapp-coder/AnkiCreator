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
# CONFIGURAÇÃO BÁSICA
# =========================
st.set_page_config(page_title="Anki-Generator", page_icon="🧠", layout="wide")

# Lê chave da OpenAI de env OU de Secrets (Streamlit Cloud)
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
"""

# =========================
# UTILITÁRIOS
# =========================
def slugify(text: str, maxlen: int = 64) -> str:
    t = unidecode(text or "").lower()
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t[:maxlen] or f"slug-{int(time.time())}"

def examples_to_html(examples: Optional[List[Dict[str, Any]]]) -> str:
    if not examples: return ""
    items = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        line = str(ex.get("text",""))
        tr = ex.get("translation"); nt = ex.get("notes")
        if tr: line += f' <span class="tr">({tr})</span>'
        if nt: line += f' <span class="nt">[{nt}]</span>'
        items.append(f"<li>{line}</li>")
    return "<ul class='examples-list'>" + "".join(items) + "</ul>"

# ---- gTTS: mapeia idioma para códigos suportados, evitando avisos
def map_lang_for_gtts(language_code: str) -> str:
    langs = tts_langs()  # {code: name}
    if not language_code:
        return "en"
    lc = language_code.lower()
    prefs = {
        "fr": ["fr"],
        "pt": ["pt", "pt-br"],
        "en": ["en", "en-us", "en-gb"],
        "es": ["es"],
        "de": ["de"],
        "it": ["it"],
        "zh": ["zh", "zh-cn", "cmn-cn", "zh-tw"],
        "ja": ["ja"],
    }
    for k, choices in prefs.items():
        if lc.startswith(k):
            for code in choices:
                if code in langs:
                    return code
    for cand in (lc, lc.replace("_","-"), lc.split("-")[0]):
        if cand in langs:
            return cand
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

# ---- TAG SANITIZER (genanki não aceita espaços nas tags)
def _clean_tag(tag) -> str:
    t = str(tag or "").strip().lower()
    t = re.sub(r"\s+", "-", t)            # espaços -> hífen
    t = re.sub(r"[^a-z0-9_\-:]", "", t)   # remove chars problemáticos
    return t[:40]                          # limite razoável

def sanitize_tags(tags) -> list:
    if not isinstance(tags, list):
        return []
    out, seen = [], set()
    for t in tags:
        ct = _clean_tag(t)
        if ct and ct not in seen:
            out.append(ct)
            seen.add(ct)
        if len(out) >= 12:
            break
    return out

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

def ingest_files(uploaded_files) -> List[Dict[str, Any]]:
    mats = []
    for f in uploaded_files or []:
        name = f.name
        data = f.read()
        low = name.lower()
        text = ""
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

def compress_materials(materials: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """Concatena trechos com cabeçalho por arquivo e corta gentilmente."""
    if not materials: return ""
    parts = []
    per = max_chars // max(1, len(materials))
    for m in materials:
        chunk = (m["content"] or "")[:per]
        parts.append(f"# {m['file']}\n{chunk}")
    return "\n\n".join(parts)[:max_chars]

# =========================
# OPENAI → GERAÇÃO DO DECK (básico)
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
    # Normalização defensiva
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception:
        data = {}

    cards = data.get("cards", [])
    if not isinstance(cards, list):
        cards = []

    # Normaliza cloze (compat Text/BackExtra)
    for c in cards:
        if not isinstance(c, dict):
            continue
        ctype = (c.get("type") or "basic").lower()
        if ctype == "cloze":
            if not c.get("front") and c.get("Text"):
                c["front"] = c["Text"]
            if not c.get("back") and c.get("BackExtra"):
                c["back"] = c["BackExtra"]

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
    """Assinatura simples para detectar duplicatas (tipo + frente + verso)."""
    t = (card.get("type") or "").lower().strip()
    f = (card.get("front") or card.get("Text") or "").strip().lower()
    b = (card.get("back") or card.get("BackExtra") or "").strip().lower()
    return f"{t}|{f[:160]}|{b[:160]}"

def dedupe_cards(cards: list) -> list:
    seen, out = set(), []
    for c in cards or []:
        if not isinstance(c, dict):
            continue
        sig = _card_signature(c)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(c)
    return out

def gerar_cartoes_adicionais(payload: dict, ja_gerados: list, faltantes: int, lote: int = 20) -> list:
    """
    Pede ao modelo APENAS os cartões faltantes (em lotes), evitando duplicatas.
    Retorna lista de cards.
    """
    novos_total = []
    restantes = max(0, int(faltantes))

    # resumo mínimo dos já gerados para reduzir duplicação
    resumo = []
    for c in ja_gerados or []:
        resumo.append({
            "type": c.get("type"),
            "front": c.get("front") or c.get("Text"),
            "back":  c.get("back")  or c.get("BackExtra"),
            "tags":  c.get("tags", [])
        })

    while restantes > 0:
        pedir = min(restantes, lote)
        sys_addendum = (
            SYSTEM_PROMPT
            + "\n\nINSTRUÇÃO ADICIONAL: Gere EXATAMENTE o número solicitado e responda SOMENTE com JSON no formato "
              '{"cards":[{...}]}'
              " (sem 'deck' e sem 'qa_report'). Não repita nenhum cartão existente."
        )
        pedido = {
            "pedido": f"Gerar exatamente {pedir} novos cartões NÃO-DUPLICADOS, mantendo active recall e conhecimento atômico.",
            "payload_base": {
                "idioma_alvo": payload.get("idioma_alvo"),
                "nivel_proficiencia": payload.get("nivel_proficiencia"),
                "topico": payload.get("topico"),
                "tipos_permitidos": payload.get("tipos_permitidos"),
                "politica_voz": payload.get("politica_voz"),
                "materiais_digest": payload.get("materiais_digest","")
            },
            "cartoes_ja_gerados_resumo": resumo
        }
        resp = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role":"system","content": sys_addendum},
                {"role":"user","content": json.dumps(pedido, ensure_ascii=False)}
            ],
            temperature=0,
            response_format={"type":"json_object"},
        )
        try:
            data = json.loads(resp.choices[0].message.content)
        except Exception:
            data = {}
        novos = data.get("cards", [])
        if not isinstance(novos, list):
            novos = []

        # normaliza cloze (compat Text/BackExtra)
        for c in novos:
            if not isinstance(c, dict):
                continue
            if (c.get("type") or "").lower() == "cloze":
                if not c.get("front") and c.get("Text"):
                    c["front"] = c["Text"]
                if not c.get("back") and c.get("BackExtra"):
                    c["back"] = c["BackExtra"]

        # junta e deduplica contra o que já temos
        candidatos = dedupe_cards((ja_gerados or []) + novos)
        have = {_card_signature(x) for x in (ja_gerados or [])}
        realmente_novos = [c for c in candidatos if _card_signature(c) not in have]

        ja_gerados = dedupe_cards((ja_gerados or []) + realmente_novos)
        novos_total += realmente_novos
        restantes = max(0, faltantes - len(novos_total))

        # se o modelo não cooperar, tenta sair do loop
        if pedir > 0 and len(realmente_novos) == 0:
            break

    return novos_total[:faltantes]

def gerar_baralho_estrito(payload: dict, max_rodadas: int = 4, lote: int = 20) -> dict:
    """
    Garante exatamente payload['limite_cartoes'].
    1) Gera o baralho base.
    2) Se vier < limite, faz rodadas complementando até alcançar.
    """
    desired = int(payload.get("limite_cartoes", 20))
    base = gerar_baralho(payload)
    cards = dedupe_cards(base.get("cards", []))

    rod = 0
    while len(cards) < desired and rod < max_rodadas:
        faltam = desired - len(cards)
        novos = gerar_cartoes_adicionais(payload, cards, faltam, lote=lote)
        cards = dedupe_cards(cards + novos)
        rod += 1

    # ajusta meta e corta ao tamanho pedido
    base["cards"] = cards[:desired]
    base.setdefault("deck", {})
    base["deck"]["card_count_planned"] = len(base["cards"])
    return base

# =========================
# ANKI (genanki)
# =========================
COMMON_CSS = """
.card { font-family: -apple-system, Segoe UI, Roboto, Arial; font-size: 20px; text-align: left; color: #222; background: #fff; }
.hint { margin-top: 8px; font-size: 0.9em; color: #666; }
.examples { margin-top: 12px; }
.examples-list { padding-left: 18px; }
.tr { color: #444; font-style: italic; }
.nt { color: #666; }
.extra { margin-top: 10px; font-size: 0.9em; color: #444; }
hr { margin: 12px 0; }
"""

MODEL_BASIC = genanki.Model(
    1607392301, "Anki-Generator Basic",
    fields=[{"name":"Front"},{"name":"Back"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Card 1",
        "qfmt":"{{Front}} {{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"{{FrontSide}}<hr id='answer'>{{Back}}"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
    }],
    css=COMMON_CSS
)

MODEL_REVERSE = genanki.Model(
    1607392302, "Anki-Generator Reverse",
    fields=[{"name":"Front"},{"name":"Back"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Reverse Only",
        "qfmt":"{{Back}} {{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"{{Back}}<hr id='answer'>{{Front}}"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
    }],
    css=COMMON_CSS
)

MODEL_CLOZE = genanki.Model(
    1607392303, "Anki-Generator Cloze",
    fields=[{"name":"Text"},{"name":"BackExtra"},{"name":"Hint"},{"name":"Examples"},{"name":"Audio"},{"name":"Extra"}],
    templates=[{
        "name":"Cloze Card",
        "qfmt":"{{cloze:Text}} {{#Hint}}<div class='hint'>{{Hint}}</div>{{/Hint}}",
        "afmt":"{{cloze:Text}}<hr id='answer'>{{BackExtra}}"
               "{{#Examples}}<div class='examples'>{{Examples}}</div>{{/Examples}}"
               "{{#Audio}}<div>{{Audio}}</div>{{/Audio}}"
               "{{#Extra}}<div class='extra'>{{Extra}}</div>{{/Extra}}"
    }],
    css=COMMON_CSS,
    model_type=genanki.Model.CLOZE
)

def build_deck_id(title: str) -> int:
    h = hashlib.sha1(title.encode("utf-8")).hexdigest()
    return int(h[:10], 16)

def build_apkg_bytes(deck_json: Dict[str, Any], tts_policy: str = "exemplos") -> bytes:
    meta = deck_json.get("deck", {})
    cards = deck_json.get("cards", [])
    title = meta.get("title","Anki-Generator Deck")
    lang  = meta.get("language","en")

    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    def add_note(c: Dict[str, Any]):
        ctype = (c.get("type") or "basic").lower()
        hint = c.get("hint") or ""
        examples_html = examples_to_html(c.get("examples"))
        extra_bits = []
        src = c.get("source_ref") or {}
        if src.get("file") or src.get("page_or_time"):
            f = src.get("file") or ""
            p = src.get("page_or_time") or ""
            extra_bits.append(f"<div><b>Fonte:</b> {f} {p}</div>")
        if c.get("rationale"):
            extra_bits.append(f"<div><b>Por quê:</b> {c['rationale']}</div>")
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
            text = c.get("front","")
            back_extra = c.get("back","") or ""
            note = genanki.Note(
                model=MODEL_CLOZE,
                fields=[text, back_extra, hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
        elif ctype == "reverse":
            note = genanki.Note(
                model=MODEL_REVERSE,
                fields=[c.get("front",""), c.get("back",""), hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
        else:
            note = genanki.Note(
                model=MODEL_BASIC,
                fields=[c.get("front",""), c.get("back",""), hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
        deck.add_note(note)

    for c in cards:
        if isinstance(c, dict):
            add_note(c)

    pkg = genanki.Package(deck)
    if media_files:
        pkg.media_files = media_files

    with tempfile.NamedTemporaryFile(delete=False, suffix=".apkg") as tmp:
        pkg.write_to_file(tmp.name)
        with open(tmp.name, "rb") as f:
            apkg_bytes = f.read()
    return apkg_bytes

# =========================
# INTERFACE STREAMLIT
# =========================
st.title("🧠 Anki-Generator")
st.caption("Gere baralhos de Anki por tema ou a partir de documentos. (Cartões seguem recordação ativa + conhecimento atômico.)")

col1, col2 = st.columns([1,2])
with col1:
    idioma = st.selectbox("Idioma", ["fr-FR","pt-BR","en-US","es-ES","zh-CN"], index=0)
    nivel  = st.text_input("Nível", "B1")
    limite = st.slider("Qtde de cartões", 4, 80, 20, 1)
    tipos  = st.multiselect("Tipos de cartão", ["basic","reverse","cloze"], default=["basic","reverse","cloze"])
    tts    = st.radio("TTS (gTTS)", ["nenhuma","respostas","exemplos","todas"], index=2)
    strict = st.checkbox("Exigir exatamente N (completar se vierem menos)", value=True)
with col2:
    topico = st.text_area("Tópico", "Tempos do passado em francês: passé composé, imparfait, plus-que-parfait, passé simple – usos, diferenças e exemplos", height=120)
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
            "materiais_digest": digest
        }
        data = gerar_baralho_estrito(payload) if strict else gerar_baralho(payload)
        apkg_bytes = build_apkg_bytes(data, tts_policy=tts)
        file_name = f"Anki-Generator_{int(time.time())}.apkg"

        st.success(f"Baralho gerado com {len(data.get('cards', []))} cartões!")
        st.download_button("⬇️ Baixar .apkg", data=apkg_bytes, file_name=file_name, mime="application/octet-stream")
        with st.expander("🔍 Ver JSON gerado (debug)"):
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
