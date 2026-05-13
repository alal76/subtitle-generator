# Video Subtitler

A local, privacy-first web app that transcribes video audio using [Whisper](https://github.com/openai/whisper), translates subtitles offline via [argostranslate](https://github.com/argosopentech/argos-translate), and generates AI-dubbed audio with per-speaker neural voices via [edge-tts](https://github.com/rany2/edge-tts).

Everything runs on your machine. No cloud API keys required.

---

## Features

| Feature | Details |
|---|---|
| **Transcription** | faster-whisper (tiny → large-v3); CPU int8 |
| **Translation** | 19 languages, fully offline after first download (~50 MB/pair) |
| **Subtitle export** | SRT · VTT · plain text |
| **Speaker diarization** | MFCC + F0 clustering; up to 4 speakers auto-detected |
| **AI dubbing** | Per-speaker neural voice, rate ±5 %, pitch ±5 Hz via edge-tts |
| **Audio muxing** | Replace original audio or add dubbed track alongside it |
| **Media info** | ffprobe stream/format details in a modal popup |
| **Activity log** | Collapsible live sidebar — timestamps, progress, errors |

---

## Requirements

| | Windows | macOS | Linux |
|---|---|---|---|
| Python | 3.9 – 3.12 | 3.9 – 3.12 | 3.9 – 3.12 |
| ffmpeg + ffprobe | winget / manual | `brew install ffmpeg` | package manager |
| Internet (first run) | argos language packs + edge-tts voices | same | same |

---

## Installation & Usage

### macOS

```bash
# 1. Install ffmpeg (if not already installed)
brew install ffmpeg

# 2. Clone the repository
git clone https://github.com/alal76/subtitle-generator.git
cd subtitle-generator

# 3. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Run the app
python app.py
```

The app opens automatically at **http://127.0.0.1:5001** in your default browser.

---

### Windows

#### Option A — Automated (recommended)

Download and double-click **`build_windows.bat`**.

It will automatically:
1. Install Git (via winget) if not present
2. Clone or update the repository
3. Install Python 3.11 (via winget) if not present
4. Install ffmpeg + ffprobe (via winget, then PowerShell fallback)
5. Create a Python virtual environment
6. Install all dependencies from `requirements.txt`
7. Build a standalone `VideoSubtitler.exe` with PyInstaller

The resulting executable is at:
```
%USERPROFILE%\subtitle-generator\dist\VideoSubtitler\VideoSubtitler.exe
```
Zip the entire `dist\VideoSubtitler\` folder to share — recipients need neither Python nor ffmpeg.

#### Option B — Run from source

```bat
:: In a command prompt or PowerShell:

:: 1. Install ffmpeg (winget)
winget install --id Gyan.FFmpeg

:: 2. Clone and enter the repo
git clone https://github.com/alal76/subtitle-generator.git
cd subtitle-generator

:: 3. Create venv and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

:: 4. Run
python app.py
```

---

### Linux

Linux is **fully supported when running from source**. The standalone PyInstaller build has not been tested on Linux but should work.

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# Fedora / RHEL
sudo dnf install ffmpeg

# Arch
sudo pacman -S ffmpeg

# Then:
git clone https://github.com/alal76/subtitle-generator.git
cd subtitle-generator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app does **not** auto-open a browser on Linux — navigate to **http://127.0.0.1:5001** manually (or pass `--no-sandbox` to Chromium if running headless).

> **Note:** `edge-tts` requires an outbound HTTPS connection to Microsoft's speech synthesis service. All other features (transcription, translation, diarization) are fully offline once models are downloaded.

---

## Workflow

```
1 — Upload video      →  select or drag-drop (MP4, MKV, AVI, MOV, WEBM, FLV, M4V, WMV)
2 — Options           →  Whisper model size, source language
3 — Translate         →  pick one or more target languages (optional)
4 — Generate          →  runs transcription + translation
5 — Results           →  view / download SRT · VTT · TXT per language
6 — Dub Audio         →  load speaker profiles → assign voices → generate MP3
                          └─ Mux into video: replace or add audio track → download
```

**Media Info** button (upload card) — shows codec, resolution, FPS, channels, bitrate via ffprobe. Available as soon as a file is selected.

**Activity Log** sidebar — collapsible live log of every step with timestamps and colour-coded status (progress / success / error).

---

## Models & Caches

All downloaded data is stored inside the project folder under `.local_cache/`:

```
.local_cache/
  huggingface/    ← Whisper model weights
  data/           ← argostranslate language packages
  video/          ← uploaded videos (kept for muxing)
  audio/          ← extracted WAV + dubbed MP3 outputs
  mux/            ← muxed video outputs
```

Delete `.local_cache/` to reclaim disk space and start fresh.

---

## Performance Notes

- **tiny / base**: transcribes ~10× real-time on modern CPU
- **small / medium**: 2–5× real-time — good accuracy
- **large-v3**: near-real-time or slower on CPU; best accuracy
- Dubbing time scales with segment count and network latency to the edge-tts service
- Speaker diarization runs entirely on CPU using librosa + scikit-learn (no GPU needed)

---

## Dependencies

```
faster-whisper>=1.0.0     # Whisper inference (CTranslate2 backend)
flask>=3.0.0              # Web UI
argostranslate==1.9.6     # Offline translation
edge-tts>=6.1.9           # Neural TTS (Microsoft Edge voices)
pydub>=0.25.1             # Audio processing & muxing
librosa>=0.10.0           # MFCC / F0 extraction for diarization
scikit-learn>=1.3.0       # Agglomerative clustering for diarization
ffmpeg                    # Audio extraction & video muxing (system binary)
```

---

## Linux Compatibility Summary

| Component | Linux status |
|---|---|
| Transcription (faster-whisper) | ✅ Full support |
| Translation (argostranslate) | ✅ Full support |
| Subtitle export | ✅ Full support |
| Speaker diarization | ✅ Full support |
| AI dubbing (edge-tts) | ✅ Requires outbound HTTPS |
| Audio muxing (ffmpeg) | ✅ Full support |
| Media info (ffprobe) | ✅ Full support |
| Auto-open browser | ⚠️ May not work on headless servers |
| PyInstaller standalone build | ⚠️ Untested (likely works; not validated) |

---

## Repository

**https://github.com/alal76/subtitle-generator**
