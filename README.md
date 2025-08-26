# 🧠 Anki-Generator

Turn any topic or document (PDF/DOCX/TXT/Markdown) into high-quality Anki decks that follow **active recall** and **atomic knowledge** principles.  
Built with **Streamlit**, **OpenAI API**, and **genanki** — optional TTS via **gTTS**.

---

## ✨ What it does

- Generate flashcards from a **topic prompt** and/or your **uploaded files**.
- Produces **Basic / Reverse / Cloze** cards with short, meaningful examples.
- Compiles a ready-to-import **`.apkg`** file.
- (Optional) Adds audio using **gTTS** for quick listening practice.
- Encourages **active recall** and **one idea per card** (atomic knowledge).

---

## 🖼️ How it works (pipeline)

1. You enter a topic and/or upload documents (PDF/DOCX/TXT/MD).
2. The app compacts long texts and sends a structured prompt to the OpenAI API.
3. The model returns a validated **JSON deck** (cards + metadata).
4. The app builds an **Anki package (`.apkg`)** with `genanki` (and audio if enabled).
5. You download and import into Anki.

---

## 🚀 Quick Start

### Option A — Deploy on **Streamlit Community Cloud** (recommended)
1. Fork or upload this repo to GitHub.
2. Go to **https://streamlit.io/cloud → New app** and select your repo.
3. In **Settings → Secrets**, add: OPENAI_API_KEY = sk-xxxxxxxxxxxxxxxx
4. Deploy. You’ll get a public URL to share with friends.

### Option B — Run locally
```bash
# 1) Create & activate a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Set your OpenAI API key
export OPENAI_API_KEY="sk-xxxxxxxx"   # Windows (PowerShell): $env:OPENAI_API_KEY="sk-xxxxxxxx"

# 4) Run
streamlit run app.py
Open the local URL shown in the terminal.

🧩 Features

Input: topic prompt + optional file uploads (PDF/DOCX/TXT/Markdown).

Cards: Basic, Reverse, Cloze (Cloze uses {{c1::...}}).

Quality rules: active recall, one idea per card, concise wording, small hints.

Audio (optional): TTS via gTTS from a model-generated audio_script.

Packaging: single .apkg download with embedded media.

🛠️ Configuration

Environment

OPENAI_API_KEY (required): set via Streamlit Secrets or env var.

Models

Default chat model: gpt-4o-mini (edit in app.py → TEXT_MODEL).

You can switch to another model available on your OpenAI account.

File formats

PDF (pypdf), DOCX (python-docx), TXT/MD.

Large files are compacted before sending to the model (see compress_materials).

TTS

Uses gTTS; language auto-mapping handles common codes (fr, pt-BR, zh, etc.).

Toggle the TTS option in the UI.

🧭 Usage Tips

Be specific in the Topic (e.g.,
“French past tenses — passé composé vs imparfait vs plus-que-parfait: usages, aspect, temporal markers, and short examples.”).

Upload source PDFs/notes to ground the deck and get source references.

Prefer many small cards to a few large ones; cloze is great for formulas/pairs.

If you want strictly N cards, increase the requested number a bit (the app normalizes counts, but the model may return fewer if the topic is very narrow).

📦 Project Structure
.
├─ app.py               # Streamlit app (UI + pipeline)
├─ requirements.txt     # Python dependencies
└─ README.md            # This file

🔐 Privacy & Safety

Never commit API keys to the repo. Use Streamlit Secrets or environment variables.
Uploaded files are processed in memory on the server where you deploy.
Avoid uploading sensitive content; review output before sharing.

🧰 Troubleshooting
“Set your API key” error: add OPENAI_API_KEY to Secrets (Cloud) or set env var locally.
gTTS warnings about zh-CN: the app maps language codes to the gTTS-supported set; warnings are harmless, but you can change the topic language to zh if needed.
Very large PDFs: split them or reduce size; the app compacts text but extremely long inputs may truncate.
Rate limits: if the OpenAI API throttles, try again or reduce deck size.

📝 Customization

Open app.py:
Adjust the instructional SYSTEM_PROMPT to your learning style.
Change the model (TEXT_MODEL).
Tweak compaction size in compress_materials.
Replace gTTS with another TTS provider if desired.

📄 License

MIT — see LICENSE (feel free to change to your preferred license).

🙏 Acknowledgments

Streamlit
OpenAI Python SDK
genanki
gTTS
pypdf
python-docx
