"""
Streamlit Anki Deck Generator (Topic or Documents) with OpenAI + TTS

Usage
- Run:  
    pip install streamlit openai genanki pypdf python-dotenv  
    streamlit run app.py
- Provide a topic OR upload documents (.txt/.md/.pdf)
- Click **Suggest plan** → **Generate deck** → **Download .apkg** for Anki

Notes on Best Practices (built-in)
- Atomic facts (one fact per card), clear phrasing, active recall
- Reverse cards only for symmetric pairs (e.g., vocab)
- Cloze for processes/definitions; natural spans (no dangling articles)
- Include short, concrete examples; disambiguate where needed
- AI suggests card counts and type mix (20–60 by default); adjustable
"""

from __future__ import annotations
import os
import io
import re
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import streamlit as st

# Third‑party
try:
    import genanki
except ImportError:
    st.stop()

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    st.error("Missing dependency: openai. Install with `pip install openai`.")
    st.stop()

# -------------------------
# Data structures
# -------------------------

@dataclass
class Card:
    card_type: str  # 'basic', 'basic_reverse', 'cloze'
    front: str
    back: str
    example: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    audio_front: Optional[str] = None  # path to mp3 (added to media)
    audio_back: Optional[str] = None

@dataclass
class DeckPlan:
    suggested_total: int
    mix: Dict[str, int]
    rationale: str
    notes: str

# -------------------------
# OpenAI helpers
# -------------------------

SYSTEM_PLANNER = (
    "You are an expert learning scientist and Anki power user. "
    "Given a learner goal and/or source text, you produce a concise deck plan: total card count, mix of Basic, Basic (and reversed), and Cloze, with rationale. "
    "Respect cognitive load: prefer 20–60 cards per focused deck unless the user explicitly asks for more. "
    "Favor Cloze for processes/definitions; Basic for direct facts; Basic+Reverse only for symmetric pairs (e.g., word <-> translation). "
    "Ensure atomicity (one fact per card) and include concrete examples where useful."
)

PLANNER_PROMPT = """
Learner profile (if given): {profile}
Deck topic: {topic}
Source text (optional, may be long):\n{source_text}

Return JSON with keys: suggested_total (int), mix (object with keys basic, basic_reverse, cloze), rationale (short), notes (short tips).
If the topic is vocabulary or short symmetric facts in a foreign language, include some basic_reverse in the mix; otherwise keep it low or zero.
Cap total between 20 and {max_cards} unless the content obviously needs more.
"""

SYSTEM_CARDWRITER = (
    "You are an elite Anki card author. You write unambiguous, atomic, high-yield cards with clear phrasing. "
    "Follow best practices: minimal information, active recall, context where necessary, avoid synonyms that make grading fuzzy. "
    "For cloze, use {{c1::...}} style with natural spans; avoid leaving dangling articles/pronouns. "
    "If asked for language learning, include one example sentence (short, natural)."
)

CARDWRITER_PROMPT = """
Create {n} Anki cards for the deck below. Use the requested type distribution. Use the given tags and keep content atomic. If vocabulary, include transliteration if relevant.

Deck topic: {topic}
Target language (if any, else None): {lang}
Types desired (approximate counts): {mix}
Constraints: {constraints}

Source material (optional):\n{source_text}

Return a **JSON object** with a top-level key `cards`, where `cards` is a list of objects. Each object has: card_type (basic|basic_reverse|cloze), front, back, example (optional), tags (list of strings). Keep strings single-line (use 
 if needed). Make sure exactly {{cN::...}} syntax for cloze.
"""

# -------------------------
# I/O helpers
# -------------------------

def read_text_from_docs(files) -> str:
    texts: List[str] = []
    for f in files or []:
        name = getattr(f, 'name', 'upload')
        ext = os.path.splitext(name)[1].lower()
        try:
            if ext in [".txt", ".md", ".csv"]:
                texts.append(f.read().decode("utf-8", errors="ignore"))
            elif ext == ".pdf" and PdfReader is not None:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(f.read())
                    tmp.flush()
                    reader = PdfReader(tmp.name)
                buf = []
                for page in reader.pages:
                    try:
                        buf.append(page.extract_text() or "")
                    except Exception:
                        continue
                texts.append("\n".join(buf))
            else:
                texts.append("")
        except Exception as e:
            st.warning(f"Could not read {name}: {e}")
    return "\n\n".join([t for t in texts if t])


def get_client(api_key: Optional[str]) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OpenAI API key not provided. Set in the sidebar or OPENAI_API_KEY env var.")
    os.environ["OPENAI_API_KEY"] = key
    return OpenAI()

# -------------------------
# Generation core
# -------------------------

def suggest_deck_plan(client: OpenAI, topic: str, source_text: str, max_cards: int, profile: str = "") -> DeckPlan:
    prompt = PLANNER_PROMPT.format(
        profile=profile,
        topic=topic,
        source_text=source_text[:6000],
        max_cards=max_cards,
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PLANNER},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    import json
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    # Normalize mix to a dict of ints
    def to_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default
    mix_raw = data.get("mix", {}) if isinstance(data, dict) else {}
    mix = {
        "basic": to_int(mix_raw.get("basic", 0)),
        "basic_reverse": to_int(mix_raw.get("basic_reverse", 0)),
        "cloze": to_int(mix_raw.get("cloze", 0)),
    }
    suggested_total = to_int(data.get("suggested_total", sum(mix.values()) or 40), 40)
    rationale = data.get("rationale", "") if isinstance(data, dict) else ""
    notes = data.get("notes", "") if isinstance(data, dict) else ""
    return DeckPlan(suggested_total=suggested_total, mix=mix, rationale=rationale, notes=notes)


def generate_cards(client: OpenAI, topic: str, source_text: str, n: int, mix: Dict[str, int], lang: str, constraints: str, tags: List[str]) -> List[Card]:
    prompt = CARDWRITER_PROMPT.format(
        n=n,
        topic=topic,
        lang=lang,
        mix=mix,
        constraints=constraints,
        source_text=source_text[:10000],
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_CARDWRITER},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    import json
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        # Sometimes models wrap JSON in a string; try a second parse
        data = json.loads(json.loads(raw))
    # Accept either an object with key 'cards' or a bare list
    items = None
    if isinstance(data, dict):
        for key in ("cards", "items", "data", "list"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                break
    if items is None and isinstance(data, list):
        items = data
    if not isinstance(items, list):
        raise ValueError("Model did not return a list of cards under 'cards'.")

    cards: List[Card] = []
    for it in items:
        if isinstance(it, str):
            # Skip stray strings; or treat as a simple 'front' with empty back
            # Here we skip to avoid malformed notes
            continue
        c = Card(
            card_type=(it.get("card_type") or "basic").strip(),
            front=(it.get("front") or "").strip(),
            back=(it.get("back") or "").strip(),
            example=((it.get("example") or "").strip() or None),
            tags=list(set((it.get("tags") or []) + tags)),
        )
        cards.append(c)
    return cards

# -------------------------
# TTS
# -------------------------

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def synthesize_tts_batch(client: OpenAI, texts: List[str], voice: str = "alloy", outdir: str = "media") -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, txt in enumerate(texts):
        clean = re.sub(r"\s+", " ", txt).strip()
        if not clean:
            paths.append("")
            continue
        filename = os.path.join(outdir, f"tts_{i:04d}.mp3")
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=clean,
        ) as resp:
            resp.stream_to_file(filename)
        paths.append(filename)
    return paths

# -------------------------
# Anki models & packaging
# -------------------------

def build_models() -> Dict[str, genanki.Model]:
    basic_model = genanki.Model(
        1607392319,
        "Basic (with example & audio)",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
            {"name": "Example"},
            {"name": "AudioFront"},
            {"name": "AudioBack"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}<br>{{#AudioFront}}<br>[sound:{{AudioFront}}]{{/AudioFront}}",
                "afmt": "{{Front}}<hr id=answer>{{Back}}<br>{{#Example}}<div style='margin-top:8px;color:#555'>Example: {{Example}}</div>{{/Example}}<br>{{#AudioBack}}<br>[sound:{{AudioBack}}]{{/AudioBack}}",
            }
        ],
        css="""
            .card { font-family: Arial; font-size: 20px; text-align: left; color: #222; }
        """,
    )

    basic_rev_model = genanki.Model(
        1121222333,
        "Basic (and reversed) w/ audio",
        fields=[
            {"name": "Front"},
            {"name": "Back"},
            {"name": "Example"},
            {"name": "AudioFront"},
            {"name": "AudioBack"},
        ],
        templates=[
            {
                "name": "Forward",
                "qfmt": "{{Front}}<br>{{#AudioFront}}<br>[sound:{{AudioFront}}]{{/AudioFront}}",
                "afmt": "{{Front}}<hr id=answer>{{Back}}<br>{{#Example}}<div style='margin-top:8px;color:#555'>Example: {{Example}}</div>{{/Example}}<br>{{#AudioBack}}<br>[sound:{{AudioBack}}]{{/AudioBack}}",
            },
            {
                "name": "Reverse",
                "qfmt": "{{Back}}<br>{{#AudioBack}}<br>[sound:{{AudioBack}}]{{/AudioBack}}",
                "afmt": "{{Back}}<hr id=answer>{{Front}}<br>{{#Example}}<div style='margin-top:8px;color:#555'>Example: {{Example}}</div>{{/Example}}<br>{{#AudioFront}}<br>[sound:{{AudioFront}}]{{/AudioFront}}",
            },
        ],
    )

    cloze_model = genanki.Model(
        998877661,
        "Cloze (with example & audio)",
        fields=[
            {"name": "Text"},
            {"name": "BackExtra"},
            {"name": "Example"},
            {"name": "Audio"},
        ],
        templates=[
            {
                "name": "Cloze",
                "qfmt": "{{cloze:Text}}<br>{{#Audio}}<br>[sound:{{Audio}}]{{/Audio}}",
                "afmt": "{{cloze:Text}}<br><hr id=answer>{{BackExtra}}<br>{{#Example}}<div style='margin-top:8px;color:#555'>Example: {{Example}}</div>{{/Example}}<br>{{#Audio}}<br>[sound:{{Audio}}]{{/Audio}}",
            }
        ],
        css="""
            .card { font-family: Arial; font-size: 20px; text-align: left; color: #222; }
        """,
        model_type=genanki.Model.CLOZE,
    )

    return {"basic": basic_model, "basic_reverse": basic_rev_model, "cloze": cloze_model}


def package_deck_bytes(deck_name: str, cards: List[Card], voice: Optional[str], tts_on: List[str]) -> bytes:
    models = build_models()
    deck = genanki.Deck(genanki.guid_for(deck_name), deck_name)
    media_files = []

    # Optional TTS
    if voice and tts_on:
        client = OpenAI()
        texts: List[str] = []
        for c in cards:
            if c.card_type == "cloze":
                if "question" in tts_on:
                    texts.append(strip_html(c.front))
                if "answer" in tts_on:
                    texts.append(strip_html(c.back or c.front))
            else:
                if "question" in tts_on:
                    texts.append(strip_html(c.front))
                if "answer" in tts_on:
                    texts.append(strip_html(c.back))
        tts_paths = synthesize_tts_batch(client, texts, voice=voice)
        idx = 0
        for c in cards:
            if "question" in tts_on:
                c.audio_front = tts_paths[idx] if idx < len(tts_paths) else None
                if c.audio_front:
                    media_files.append(c.audio_front)
                idx += 1
            if "answer" in tts_on:
                c.audio_back = tts_paths[idx] if idx < len(tts_paths) else None
                if c.audio_back:
                    media_files.append(c.audio_back)
                idx += 1

    for c in cards:
        if c.card_type == "cloze":
            model = models["cloze"]
            fields = [c.front, c.back or "", c.example or "", (os.path.basename(c.audio_front or c.audio_back) if (c.audio_front or c.audio_back) else "")]
            note = genanki.Note(model=model, fields=fields, tags=c.tags)
        elif c.card_type == "basic_reverse":
            model = models["basic_reverse"]
            fields = [c.front, c.back, c.example or "", os.path.basename(c.audio_front or ""), os.path.basename(c.audio_back or "")]
            note = genanki.Note(model=model, fields=fields, tags=c.tags)
        else:
            model = models["basic"]
            fields = [c.front, c.back, c.example or "", os.path.basename(c.audio_front or ""), os.path.basename(c.audio_back or "")]
            note = genanki.Note(model=model, fields=fields, tags=c.tags)
        deck.add_note(note)

    pkg = genanki.Package(deck)
    if media_files:
        pkg.media_files = media_files
    # Write to BytesIO
    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        pkg.write_to_file(tmp.name)
        tmp.flush()
        data = open(tmp.name, "rb").read()
    return data

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Anki Deck Generator", page_icon="🧠", layout="centered")

st.title("🧠 Anki Deck Generator")
st.caption("Topic or documents → AI‑planned deck → atomic cards → optional TTS → .apkg download")

with st.sidebar:
    st.header("🔑 API & Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Not stored; used only for this session. Or set OPENAI_API_KEY env var.")
    default_model = "gpt-4o-mini"
    st.markdown("**Model**: gpt-4o-mini (fixed in this demo)")

    st.divider()
    st.header("🗣️ TTS (optional)")
    use_tts = st.checkbox("Enable TTS", value=False)
    voice = st.selectbox("Voice", ["alloy", "verse", "florence", "aria"], index=0, disabled=not use_tts)
    tts_on_q = st.checkbox("Attach audio to question", value=True, disabled=not use_tts)
    tts_on_a = st.checkbox("Attach audio to answer", value=False, disabled=not use_tts)

st.subheader("1) Input")
colA, colB = st.columns(2)
with colA:
    deck_name = st.text_input("Deck name", value="My AI Deck")
    topic = st.text_area("Topic / goal (or leave blank if using documents)", placeholder="e.g., HSK1 Chinese food vocabulary" )
    profile = st.text_input("Learner profile (optional)", placeholder="e.g., beginner Chinese, 20 new/day")
    lang = st.text_input("Target language (optional)", placeholder="e.g., zh, fr, es")
with colB:
    docs = st.file_uploader("Upload documents (.txt/.md/.pdf)", type=["txt","md","pdf"], accept_multiple_files=True)
    max_cards = st.slider("Max cards (AI suggestion cap)", 20, 500, 100, step=10)
    constraints = st.text_area("Constraints / tips to the card writer", value="Prefer atomic facts; avoid ambiguity; include examples where helpful.")

source_text = read_text_from_docs(docs) if docs else ""

if not topic and not source_text:
    st.info("Provide a topic or upload at least one document.")

st.subheader("2) Suggest plan")
if st.button("Suggest plan", disabled=not (topic or source_text)):
    try:
        client = get_client(api_key)
        with st.status("Asking AI for a deck plan…", state="running") as s:
            plan = suggest_deck_plan(client, topic=topic or (docs[0].name if docs else "Untitled"), source_text=source_text, max_cards=max_cards, profile=profile)
            s.update(label="Plan ready", state="complete")
        st.session_state["plan"] = plan
    except Exception as e:
        st.error(f"Planning failed: {e}")

plan: Optional[DeckPlan] = st.session_state.get("plan")
if plan:
    st.success(f"Suggested total: {plan.suggested_total} | Mix: {plan.mix}")
    with st.expander("Rationale & notes"):
        st.write(plan.rationale or "—")
        if plan.notes:
            st.info(plan.notes)
    # Optional warning for big TTS runs
    if use_tts and int(getattr(plan, "suggested_total", 0)) > 250:
        st.warning("TTS for very large decks (>250 cards) can take a long time and create large .apkg files. Consider turning off TTS or generating in chunks.")
    # Allow user override
    st.subheader("Adjust plan (optional)")
    total_override = st.number_input("Total cards", min_value=10, max_value=500, value=int(plan.suggested_total))
    basic = st.number_input("Basic", min_value=0, max_value=500, value=int(plan.mix.get("basic", 0)))
    basic_rev = st.number_input("Basic (and reversed)", min_value=0, max_value=500, value=int(plan.mix.get("basic_reverse", 0)))
    cloze = st.number_input("Cloze", min_value=0, max_value=500, value=int(plan.mix.get("cloze", 0)))

    if basic + basic_rev + cloze != total_override:
        st.warning("Sum of types doesn't match total. The generator will use the total value for the number of cards.")

    st.subheader("3) Generate deck")
    if st.button("Generate deck", use_container_width=True):
        try:
            client = get_client(api_key)
            tags = [re.sub(r"\W+", "_", deck_name.lower()).strip("_")]

            with st.status("Creating cards…", state="running") as s:
                cards = generate_cards(
                    client,
                    topic=(topic or deck_name),
                    source_text=source_text,
                    n=int(total_override),
                    mix={"basic": int(basic), "basic_reverse": int(basic_rev), "cloze": int(cloze)},
                    lang=lang,
                    constraints=constraints,
                    tags=tags,
                )
                s.update(label="Packaging deck…")
                tts_on = (["question"] if st.session_state.get("tts_on_q", tts_on_q) else []) + (["answer"] if st.session_state.get("tts_on_a", tts_on_a) else [])
                voice_sel = voice if use_tts else None
                apkg_bytes = package_deck_bytes(deck_name, cards, voice=voice_sel, tts_on=tts_on)
                s.update(label="Done", state="complete")

            st.session_state["apkg"] = apkg_bytes
            st.session_state["cards_preview"] = cards[: min(10, len(cards))]
            st.success("Deck generated!")
        except Exception as e:
            st.error(f"Generation failed: {e}")

# Preview & download
apkg = st.session_state.get("apkg")
preview = st.session_state.get("cards_preview")
if preview:
    st.subheader("Preview (first 10 cards)")
    for i, c in enumerate(preview, 1):
        with st.expander(f"{i}. {c.card_type.upper()} — {c.front[:60]}…"):
            st.markdown(f"**Front:** {c.front}")
            st.markdown(f"**Back:** {c.back}")
            if c.example:
                st.markdown(f"**Example:** {c.example}")
            if c.tags:
                st.caption("Tags: " + ", ".join(sorted(set(c.tags))))

if apkg:
    st.subheader("4) Download")
    st.download_button("Download .apkg", data=apkg, file_name=f"{re.sub(r'[^A-Za-z0-9_-]+','_', (deck_name or 'deck'))}.apkg", mime="application/octet-stream")

st.markdown("---")
with st.expander("ℹ️ Tips for best learning outcomes"):
    st.markdown(
        "- Keep decks focused (20–60 new facts); avoid encyclopedic mixes.\n"
        "- Prefer **atomic** cards; split complex ideas.\n"
        "- Use **Cloze** for processes/definitions; **Basic+Reverse** only for symmetric pairs.\n"
        "- Add short examples; avoid fuzzy synonyms; include disambiguation in parentheses.\n"
        "- Interleave study; tag consistently; adjust your daily new card limit to protect review health."
    )
