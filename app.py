"""
Anki Deck Generator (Topic or Documents) with OpenAI + TTS

Features
- Suggests how many cards to create and which card types to prefer per deck using AI
- Generates high‑quality, atomic Anki cards (Basic, Basic (and reversed), Cloze)
- Optional text‑to‑speech (TTS) audio on the question/answer using OpenAI TTS
- Ingests: a topic prompt OR one/more documents (.txt/.md/.pdf)
- Exports a self‑contained .apkg you can import into Anki

Requirements (install first)
    pip install openai genanki pypdf python-dotenv

Environment
    Set OPENAI_API_KEY in your environment or a .env file at project root.

Usage examples
    python anki_deck_generator.py --deck "HSK1 - Food" --topic "Chinese food vocabulary: 50 items with example sentences" --lang zh
    python anki_deck_generator.py --deck "EU Law - Asylum CEAS" --docs notes1.pdf ceas_overview.txt --max-cards 80
    python anki_deck_generator.py --deck "French Past Tenses" --topic "Passé composé vs imparfait: signals, conjugations, pitfalls" --voice alloy --tts-on question answer

Notes on best practices baked in
- Atomic facts; one fact per card; avoid lists-on-a-card
- Active recall first; optional reverse only when symmetric (e.g., vocabulary)
- Use Cloze for processes/definitions with natural blanks (no orphaned clozes)
- Include a concrete example or minimal pair where helpful
- Keep phrasing unambiguous; add disambiguating context in parentheses when needed
- Avoid synonyms that make grading fuzzy; include the canonical expected answer
- Tag cards consistently for filtering and interleaving (topic/subtopic/source)
- Keep card counts within a daily review budget; we auto-suggest a plan
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Third‑party
try:
    import genanki
except ImportError:
    print("Missing dependency: genanki. Install with `pip install genanki`.", file=sys.stderr)
    raise

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
    raise

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
    mix: Dict[str, int]  # e.g., {"basic": 20, "basic_reverse": 15, "cloze": 15}
    rationale: str
    notes: str

# -------------------------
# Helpers: I/O and ingestion
# -------------------------

def read_text_from_docs(paths: List[str]) -> str:
    texts = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[warn] File not found: {p}", file=sys.stderr)
            continue
        ext = os.path.splitext(p)[1].lower()
        try:
            if ext in [".txt", ".md", ".csv"]:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
            elif ext == ".pdf" and PdfReader is not None:
                reader = PdfReader(p)
                buf = []
                for page in reader.pages:
                    try:
                        buf.append(page.extract_text() or "")
                    except Exception:
                        continue
                texts.append("\n".join(buf))
            else:
                print(f"[warn] Unsupported extension or pypdf missing: {p}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] Could not read {p}: {e}", file=sys.stderr)
    return "\n\n".join(texts)

# -------------------------
# OpenAI wrappers
# -------------------------

def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Export it or create a .env file.", file=sys.stderr)
        sys.exit(1)
    return OpenAI()

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

Return JSON list where each item has: card_type (basic|basic_reverse|cloze), front, back, example (optional), tags (list of strings). Keep strings single-line (use \n if needed). Make sure exactly {{cN::...}} syntax for cloze.
"""

# -------------------------
# Planning and generation
# -------------------------

def suggest_deck_plan(client: OpenAI, topic: str, source_text: str, max_cards: int, profile: str = "") -> DeckPlan:
    prompt = PLANNER_PROMPT.format(
        profile=profile,
        topic=topic,
        source_text=source_text[:6000],
        max_cards=max_cards,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": SYSTEM_PLANNER},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    import json
    data = json.loads(resp.choices[0].message.content)
    suggested_total = int(data.get("suggested_total", 40))
    mix = data.get("mix", {"basic": suggested_total, "basic_reverse": 0, "cloze": 0})
    rationale = data.get("rationale", "")
    notes = data.get("notes", "")
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
    items = json.loads(resp.choices[0].message.content)
    cards: List[Card] = []
    for it in items:
        c = Card(
            card_type=it.get("card_type", "basic"),
            front=it.get("front", "").strip(),
            back=it.get("back", "").strip(),
            example=(it.get("example") or "").strip() or None,
            tags=list(set((it.get("tags") or []) + tags)),
        )
        cards.append(c)
    return cards

# -------------------------
# TTS synthesis
# -------------------------

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
            format="mp3",
        ) as resp:
            resp.stream_to_file(filename)
        paths.append(filename)
    return paths

# -------------------------
# Anki models and packaging
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


def package_deck(deck_name: str, cards: List[Card], voice: Optional[str], tts_on: List[str], out_path: str) -> str:
    models = build_models()
    deck = genanki.Deck(genanki.guid_for(deck_name), deck_name)
    media_files = []

    # If TTS enabled, synthesize first to map onto cards deterministically
    if voice and tts_on:
        client = get_client()
        texts = []
        # Collect in the same order we will assign
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
        # Re-assign into cards
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

    # Add notes
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
    pkg.write_to_file(out_path)
    return out_path

# -------------------------
# Utilities
# -------------------------

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()

# -------------------------
# Main routine
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Anki deck from topic or documents using OpenAI + TTS")
    parser.add_argument("--deck", required=True, help="Deck name")
    parser.add_argument("--topic", default="", help="Topic/goal prompt for the deck")
    parser.add_argument("--docs", nargs="*", default=[], help="Path(s) to .txt/.md/.pdf to ingest")
    parser.add_argument("--profile", default="", help="Learner profile (e.g., 'beginner Chinese, daily 20 cards')")
    parser.add_argument("--lang", default="", help="Target language (e.g., 'zh', 'fr'). Leave empty for general topics")
    parser.add_argument("--max-cards", type=int, default=60, help="Upper bound for suggested cards")
    parser.add_argument("--min-cards", type=int, default=20, help="Lower bound for fallback if suggestion fails")
    parser.add_argument("--constraints", default="Prefer atomic facts; avoid ambiguity; include examples where helpful.")
    parser.add_argument("--voice", default="", help="OpenAI TTS voice (e.g., 'alloy'). Empty disables TTS")
    parser.add_argument("--tts-on", nargs="*", default=[], choices=["question", "answer"], help="Where to attach TTS audio")
    parser.add_argument("--tags", nargs="*", default=[], help="Additional tags to attach to all cards")
    parser.add_argument("--out", default="deck.apkg", help="Output .apkg path")

    args = parser.parse_args()

    if not args.topic and not args.docs:
        print("Provide --topic or --docs (or both).", file=sys.stderr)
        sys.exit(2)

    client = get_client()

    source_text = read_text_from_docs(args.docs) if args.docs else ""
    topic = args.topic or (os.path.basename(args.docs[0]) if args.docs else "Untitled Deck")

    # 1) Suggest a plan
    plan = suggest_deck_plan(client, topic=topic, source_text=source_text, max_cards=args.max_cards, profile=args.profile)
    print("Suggested plan:")
    print(f"  Total: {plan.suggested_total}")
    print(f"  Mix: {plan.mix}")
    print(f"  Rationale: {plan.rationale}")
    if plan.notes:
        print(f"  Notes: {plan.notes}")

    # Ensure within bounds
    n = max(args.min_cards, min(args.max_cards, plan.suggested_total))

    # 2) Generate cards
    # Smart default tags
    auto_tags = [re.sub(r"\W+", "_", args.deck.lower()).strip("_")]
    tags = sorted(set(auto_tags + args.tags))

    cards = generate_cards(
        client,
        topic=topic,
        source_text=source_text,
        n=n,
        mix=plan.mix,
        lang=args.lang,
        constraints=args.constraints,
        tags=tags,
    )

    # 3) Package deck (+ optional TTS)
    voice = args.voice.strip() or None
    tts_on = args.tts_on
    out_path = os.path.abspath(args.out)
    package_deck(args.deck, cards, voice=voice, tts_on=tts_on, out_path=out_path)

    print(f"\nDone. Deck written to: {out_path}")


if __name__ == "__main__":
    main()
