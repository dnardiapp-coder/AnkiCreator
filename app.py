import os, io, json, time, tempfile, hashlib, re, random
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
# CONFIG GERAL
# =========================
st.set_page_config(page_title="Anki-Generator", page_icon="🧠", layout="wide")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("Defina OPENAI_API_KEY nos Secrets (Streamlit) ou variável de ambiente.")
    st.stop()

# Client com timeout para não travar
client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0)

# Robustez
TEXT_MODEL = "gpt-4o-mini"   # troque se quiser
STRICT_MAX_ROUNDS = 4        # rodadas de complemento
STRICT_BATCH = 20            # cartões por rodada
STRICT_HARD_TIMEOUT = 180    # seg totais no modo estrito
MAX_AUDIO_FILES = 24         # limite de áudios por deck
AUDIO_CHAR_LIMIT = 400       # limite de chars no TTS

SYSTEM_PROMPT = """
Você é uma IA especialista em Design Instrucional e Ciência Cognitiva, integrada ao 'Anki-Generator'.

OBJETIVO
- Gerar cartões úteis, específicos e de alta qualidade, ancorados no tópico e nos materiais do usuário (quando houver).
- Práticas baseadas em evidência: recordação ativa, conhecimento atômico, exemplos concretos, interleaving/variação, contraste/erro comum, foco em transferência.

FINS (payload.goal)
- "General learning": equilíbrio entre definição, cloze, cenários e procedimentos.
- "Org policy mastery": foque itens práticos: seções/IDs, prazos, thresholds/valores, exceções, responsáveis, aprovações, sanções, conformidade, auditoria, SLA/OLA. Muitos cenários e procedimentos.
- "Exam prep": foco em pontos cobrados/pegadinhas/variações; >=30% cloze, >=20% cenários; inclua erros comuns.
- "Language: Vocabulary": collocations, regência/partículas, falsos cognatos; cloze de palavras, exemplos bilíngues, frase de uso; audio_script apropriado.
- "Language: Grammar & Patterns": contraste de padrões, cloze morfossintático, condições e restrições; exemplos mínimos; audio_script com frases-alvo.
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
13) Pistas mínimas somente quando necessárias (hint curto). 14) Estruture em camadas (do básico ao complexo).
15) Quebre assuntos grandes em subcartões. 16) Não confunda reconhecimento com recall.
17) Cloze com sabedoria (um alvo por cloze). 18) Refine cartões ruins (indique notas no qa_report).
19) Personalize aos objetivos do usuário (payload.goal). 20) Teste para uso real (contextos/autênticos).

Parte 2 – Técnicas
- Basic Q&A: melhor para fatos simples (curtos).
- Cloze deletion: palavra-chave/termo/número/cláusula em contexto.
- Reverse cards: opcional quando reforça bidirecionalidade.
- Step-by-step process: um passo por cartão (ordem/quem faz/prazo).
- Component/part: para listas, teste itens individualmente.
- Cause-effect: “Se X, então Y” (resultados, condições).
- Comparison/contrast: diferenças essenciais (diagnóstico).

Parte 3 – Erros a evitar
- Cartões carregados; perguntas vagas; múltiplas respostas; cópia literal sem compreensão;
- Múltipla escolha em excesso; não editar cartões ruins; contexto demais; redação confusa.

Parte 4 – Modelos de boa pergunta
- “O que é X?” (definição); “X é Y?”/“X tem quantos Y?” (fato simples);
- “Se X, então Y?” (causa-efeito); “Por que X?” (causal);
- “Quais componentes de X?” (listas → dividir); “O que acontece em X?” (processo).

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
    """Remove cercas e converte \\n em <br>. Se cloze=True, preserva {{c1::...}}."""
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

# gTTS mapping
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

# genanki tags
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

# --- heurística Q/A & rótulos ---
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
    """Pergunta → Resposta; se nada parecer pergunta, lado mais curto é a frente."""
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
# GOAL → mix minima
# =========================
def goal_mix_minima(goal: str) -> Dict[str, float]:
    g = (goal or "").lower()
    # percentuais mínimos do conjunto atual (ajustados por boas práticas)
    if "policy" in g:
        return {"cloze": 0.25, "scenario": 0.35, "procedure": 0.25}
    if "exam prep" in g:
        return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}
    if "vocabulary" in g:
        return {"cloze": 0.35, "scenario": 0.05, "procedure": 0.00}
    if "grammar" in g or "patterns" in g:
        return {"cloze": 0.50, "scenario": 0.10, "procedure": 0.00}
    if "listening" in g or "pronunciation" in g:
        return {"cloze": 0.30, "scenario": 0.10, "procedure": 0.00}
    if "reading" in g or "cefr" in g:
        return {"cloze": 0.40, "scenario": 0.10, "procedure": 0.00}
    # General learning default
    return {"cloze": 0.30, "scenario": 0.20, "procedure": 0.15}

# =========================
# OPENAI → GERAÇÃO (básico)
# =========================
def _safe_json(content: str) -> dict:
    try:
        return json.loads(content)
    except Exception:
        return {}

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
    data = _safe_json(resp.choices[0].message.content or "{}")
    cards = data.get("cards", [])
    if not isinstance(cards, list): cards = []

    # Normalização
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
# VARIEDADE/ÂNCORA: keywords & filtros
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

def enforce_mix_and_anchoring(cards: List[dict], kws: List[str], require_anchor: bool, max_qa_frac: float, minima_pct: Dict[str,float]) -> (List[dict], dict):
    """Filtra genéricos e não ancorados; limita Q&A; calcula necessidades por tipo."""
    if not isinstance(cards, list): cards = []
    # 1) filtro
    filtered = []
    for c in cards:
        if require_anchor and not anchored_to_source(c):
            continue
        if not contains_domain_keyword(c, kws):
            continue
        filtered.append(c)

    if not filtered:
        return [], {"cloze_min":0,"scenario_min":0,"procedure_min":0}

    # 2) limitar Q&A
    total = max(1, len(filtered))
    qa_cards = [c for c in filtered if card_kind(c) == "qa"]
    max_qa = int(max_qa_frac * total)
    if len(qa_cards) > max_qa:
        qa_sorted = sorted(qa_cards, key=lambda c: len(c.get("front","")))
        to_remove = set(id(x) for x in qa_sorted[max_qa:])
        filtered = [c for c in filtered if id(c) not in to_remove]
        total = max(1, len(filtered))

    # 3) necessidades por tipo (percentuais mínimos)
    kinds = [card_kind(c) for c in filtered]
    need = {
        "cloze_min": max(0, int(minima_pct.get("cloze",0)*total) - kinds.count("cloze")),
        "scenario_min": max(0, int(minima_pct.get("scenario",0)*total) - kinds.count("scenario")),
        "procedure_min": max(0, int(minima_pct.get("procedure",0)*total) - kinds.count("procedure")),
    }
    return filtered, need

# =========================
# DEDUPE + TOP-UP (modo estrito com TIMEOUT)
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

def gerar_cartoes_adicionais(payload: dict, ja_gerados: list, faltantes: int, lote: int = STRICT_BATCH) -> list:
    novos_total = []; restantes = max(0, int(faltantes))

    resumo = []
    for c in ja_gerados or []:
        resumo.append({"type": c.get("type"),
                       "front": c.get("front") or c.get("Text"),
                       "back":  c.get("back")  or c.get("BackExtra"),
                       "tags":  c.get("tags", [])})

    pedir = min(restantes, lote)
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
        "pedido": f"Gerar exatamente {pedir} novos cartões NÃO-DUPLICADOS, variados (cloze/cenário/procedimento/definição-criterial), mantendo active recall e atômicos.",
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
            "mix_targets": payload.get("mix_targets", {})
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
    data = _safe_json(resp.choices[0].message.content or "{}")
    novos = data.get("cards", [])
    if not isinstance(novos, list): novos = []

    # normaliza
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

    return novos[:faltantes]

def gerar_baralho_estrito(payload: dict, progress=None) -> dict:
    start = time.time()
    desired = int(payload.get("limite_cartoes", 20))
    base = gerar_baralho(payload)
    cards = dedupe_cards(base.get("cards", []))

    # Primeira filtragem e metas por tipo
    kws = payload.get("domain_keywords", [])
    minima_pct = goal_mix_minima(payload.get("goal","General learning"))
    cards, need = enforce_mix_and_anchoring(cards, kws, payload.get("require_anchor", True), payload.get("max_qa_pct", 0.5), minima_pct)

    if progress: progress.progress(min(0.1, len(cards)/max(1, desired)))
    rounds = 0

    while len(cards) < desired and rounds < STRICT_MAX_ROUNDS:
        if time.time() - start > STRICT_HARD_TIMEOUT:
            break
        faltam = desired - len(cards)
        payload["mix_targets"] = need
        novos = gerar_cartoes_adicionais(payload, cards, faltam, lote=STRICT_BATCH)

        prev = len(cards)
        draft = dedupe_cards(cards + novos)
        draft, need = enforce_mix_and_anchoring(draft, kws, payload.get("require_anchor", True), payload.get("max_qa_pct", 0.5), minima_pct)
        cards = draft

        rounds += 1
        if progress:
            frac = min(0.95, len(cards) / max(1, desired) * 0.9 + 0.05)
            progress.progress(frac)
        if len(cards) == prev:
            break

    base["cards"] = cards[:desired]
    base.setdefault("deck", {})
    base["deck"]["card_count_planned"] = len(base["cards"])
    if progress: progress.progress(1.0)
    return base

# =========================
# ANKI (genanki) — MODELOS v3
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

def build_apkg_bytes(deck_json: Dict[str, Any], tts_policy: str = "exemplos", extra_kind: str = "usage_tip") -> bytes:
    meta = deck_json.get("deck", {})
    cards = deck_json.get("cards", [])
    title = meta.get("title","Anki-Generator Deck")
    lang  = meta.get("language","en")

    deck = genanki.Deck(build_deck_id(title), title)
    media_files: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="anki_media_")

    # TTS: escolha aleatória até MAX_AUDIO_FILES cartões
    idxs_for_audio = set()
    if tts_policy != "nenhuma" and len(cards) > 0:
        idxs_for_audio = set(random.sample(range(len(cards)), min(MAX_AUDIO_FILES, len(cards))))

    def add_note(c: Dict[str, Any], index: int):
        ctype = (c.get("type") or "basic").lower()
        if ctype == "reverse": ctype = "basic"  # render como basic

        hint = html_escape(c.get("hint") or "")
        examples_html = examples_to_html(c.get("examples"))

        # EXTRA
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

        # ÁUDIO
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

        # CAMPOS VISUAIS
        if ctype == "cloze":
            text = c.get("front","")
            back_extra = c.get("back","") or ""
            note = genanki.Note(
                model=MODEL_CLOZE,
                fields=[text, back_extra, hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
        else:
            front = normalize_text_for_html(front_raw)
            back  = normalize_text_for_html(back_raw)
            note = genanki.Note(
                model=MODEL_BASIC,
                fields=[front, back, hint, examples_html, audio_field, extra],
                tags=sanitize_tags(c.get("tags", []))
            )
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
# UI
# =========================
st.title("🧠 Anki-Generator")
st.caption("Cartões com active recall + conhecimento atômico. Variedade de formatos e ancoragem em documentos.")

col1, col2 = st.columns([1,2])
with col1:
    idioma = st.selectbox("Idioma", ["fr-FR","pt-BR","en-US","es-ES","zh-CN"], index=0)
    nivel  = st.text_input("Nível/CEFR (opcional)", "B1")
    limite = st.slider("Qtde de cartões", 4, 120, 24, 1)
    tipos  = st.multiselect("Tipos de cartão", ["basic","reverse","cloze"], default=["basic","reverse","cloze"])
    tts    = st.radio("TTS (gTTS)", ["nenhuma","respostas","exemplos","todas"], index=2)
    strict = st.checkbox("Exigir exatamente N (completar se vierem menos)", value=True)
    # Objetivos
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
    max_qa_pct = st.slider("Max % of Q&A cards", 20, 80, 45, 5)
    require_anchor = st.checkbox("Require source anchoring (from uploaded docs)", value=True)
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
    topico = st.text_area(
        "Tópico (ou diretrizes de geração)",
        "Tempos do passado em francês: passé composé, imparfait, plus-que-parfait, passé simple – usos, diferenças e exemplos",
        height=140
    )
    files  = st.file_uploader(
        "Arquivos (PDF/DOCX/TXT/MD – opcional; para políticas, suba os documentos oficiais)",
        type=["pdf","docx","txt","md","markdown"], accept_multiple_files=True
    )

btn = st.button("Gerar baralho (.apkg)", type="primary", use_container_width=True)

if btn:
    if not topico.strip():
        st.error("Preencha o campo Tópico.")
        st.stop()

    with st.spinner("Gerando cartões…"):
        materials = ingest_files(files) if files else []
        digest = compress_materials(materials) if materials else ""

        # Desativa TTS para decks muito grandes (rápido)
        effective_tts = tts
        if tts != "nenhuma" and limite > 40:
            effective_tts = "nenhuma"
            st.info("TTS desativado automaticamente para decks grandes (>40) para acelerar. Gere um deck menor se quiser áudio.")

        # keywords do domínio a partir dos docs (ou do tópico)
        domain_kws = extract_domain_keywords(digest) if digest else extract_domain_keywords(topico)

        payload = {
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
            "domain_keywords": domain_kws
        }

        prog = st.progress(0.0)
        data = gerar_baralho_estrito(payload, progress=prog) if strict else gerar_baralho(payload)
        apkg_bytes = build_apkg_bytes(data, tts_policy=effective_tts, extra_kind=extra_kind)
        file_name = f"Anki-Generator_{int(time.time())}.apkg"

        st.success(f"Baralho gerado com {len(data.get('cards', []))} cartões!")
        st.download_button("⬇️ Baixar .apkg", data=apkg_bytes, file_name=file_name, mime="application/octet-stream")
        with st.expander("🔍 Ver JSON gerado (debug)"):
            st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")

