# Video Subtitler

A local, privacy-first web app that transcribes video audio using [Whisper](https://github.com/openai/whisper), translates subtitles offline via [argostranslate](https://github.com/argosopentech/argos-translate), and generates AI-dubbed audio with per-speaker neural voices via [edge-tts](https://github.com/rany2/edge-tts).

Everything runs on your machine. No cloud API keys required.

---

## Features

| Feature | Details |
|---|---|
| **Transcription** | faster-whisper (tiny → large-v3); GPU (CUDA float16) when available, CPU int8 fallback |
| **Translation** | 19 languages, fully offline after first download (~50 MB/pair) |
| **Subtitle export** | SRT · VTT · plain text; auto-saved to a configured output folder |
| **Speaker diarization** | MFCC + F0 clustering; silhouette-scored — detects as many speakers as the video contains |
| **AI dubbing** | Per-speaker neural voice; speed & pitch auto-suggested from acoustic analysis; rate ±20 %, pitch ±10 Hz |
| **Bitrate matching** | Dubbed audio exported at the same bitrate as the original audio stream |
| **Video bundling** | MP4 or MKV output; choose original track, dubbed track, or both as separate streams |
| **Auto-save** | Subtitles, dubbed audio, and bundled video saved automatically to any configured folder |
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

## Quick Install (one command)

Each script clones the repo, installs all system and Python dependencies, and launches the app automatically.

### macOS

```bash
curl -fsSL https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.sh | bash
```

### Linux

```bash
curl -fsSL https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.sh | bash
```

### Windows (PowerShell — run as your normal user)

```powershell
irm https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.ps1 | iex
```

> **Custom install directory** — set `VIDEO_SUBTITLER_DIR` before running:  
> macOS/Linux: `VIDEO_SUBTITLER_DIR=~/apps/subtitler curl … | bash`  
> Windows: `$env:VIDEO_SUBTITLER_DIR="C:\apps\subtitler"; irm … | iex`

> **Custom port** — set `PORT=8080` (macOS/Linux) or `$env:PORT="8080"` (Windows) the same way.

---

## Installation & Usage

### macOS

**One-liner (recommended):**
```bash
curl -fsSL https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.sh | bash
```

**Manual steps:**
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

#### Option A — One-liner (PowerShell)

```powershell
irm https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.ps1 | iex
```

Installs Git, Python 3.11, and ffmpeg automatically via winget, then clones the repo, sets up a virtual environment, and launches the app.

#### Option B — Automated standalone build

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

#### Option C — Run from source

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

**One-liner (recommended):**
```bash
curl -fsSL https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.sh | bash
```
Auto-detects your package manager (apt / dnf / pacman / zypper) to install ffmpeg, then clones the repo and launches the app.

**Manual steps:**
```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# Fedora / RHEL
sudo dnf install ffmpeg

# Arch
sudo pacman -S ffmpeg

# openSUSE
sudo zypper install ffmpeg

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
0 — Output Settings  →  set an absolute folder path for auto-save (optional)
                         subtitles, dubbed audio, and bundled video are written there
1 — Upload video     →  select or drag-drop (MP4, MKV, AVI, MOV, WEBM, FLV, M4V, WMV)
                         Media Info button shows codec / resolution / bitrate via ffprobe
2 — Options          →  Whisper model size, source language
3 — Translate        →  pick one or more target languages (optional)
4 — Generate         →  transcription → speaker diarization → translation
5 — Results          →  view / download SRT · VTT · TXT per language
                         files also saved to output folder if configured
6 — Dub Audio        →  Load Speaker Profiles → per-speaker voice, speed, pitch
                         speed & pitch pre-filled from acoustic analysis of each speaker
                         Generate dubbed audio (bitrate matched to source)
                         └─ Bundle into video:
                              Container:    MP4 or MKV
                              Audio tracks: ☑ Keep original   ☑ Include dubbed
                              → Download bundled video
                              → File also saved to output folder if configured
```

**Activity Log** sidebar — collapsible live log of every step with timestamps and colour-coded status (progress / success / error).

---

## Output Folder & Auto-Save

The **Output Settings** card at the top of the UI accepts any absolute path. Once set, the app automatically saves:

| File | Naming convention |
|---|---|
| Subtitles | `{source_filename}_{lang}.srt` / `.vtt` / `.txt` |
| Dubbed audio | `{source_filename}_{lang}_dubbed.mp3` |
| Bundled video | `{source_filename}_bundled.mp4` (or `.mkv`) |

All files are also available for manual download from the UI. Leave the field blank to disable auto-save.

The setting persists for the session and is restored on page reload (`GET /config`).

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

## Speaker Diarization & Voice Profiles

Diarization runs entirely on CPU using MFCC features and F0 (fundamental frequency) extraction via librosa.

- **Speaker count** is detected automatically using silhouette scoring — no hard limit. A 2-hour lecture can produce 8+ distinct speakers if the signal supports it.
- **Gender** is inferred from median F0 (> 165 Hz → Female), which pre-filters the voice dropdown to matching-gender options.
- For each speaker the app computes **speaking rate** (words/second) and **median pitch** (Hz) and maps them to suggested TTS `rate` and `pitch` offsets. These pre-fill the sliders — you can adjust freely before generating.
- Sliders range: rate −20 % to +20 %, pitch −10 Hz to +10 Hz.

---

## Audio Bitrate Matching

When the source video is processed, ffprobe reads the original audio stream bitrate. The dubbed MP3 and the bundled AAC track are both encoded at that same bitrate (clamped 64 – 320 kbps, defaulting to 192 kbps if detection fails).

---

## Video Bundling

The **Bundle Options** panel (step 6) offers:

| Option | Effect |
|---|---|
| **Container** | MP4 (broad compatibility) or MKV (multi-track, subtitles) |
| **Keep original audio track(s)** | All source audio streams are copied as-is |
| **Include dubbed audio track** | TTS audio added as an additional stream |

Selecting both audio options produces a file with both tracks (e.g. English original + Spanish dub), letting media players switch between them.

---

## Performance Notes

- **tiny / base**: transcribes ~10× real-time on modern CPU
- **small / medium**: 2–5× real-time — good accuracy
- **large-v3**: near-real-time or slower on CPU; best accuracy
- Dubbing time scales with segment count and network latency to the edge-tts service
- Speaker diarization runs entirely on CPU using librosa + scikit-learn (no GPU needed)

---

## GPU Acceleration

Whisper transcription automatically uses an **NVIDIA GPU** when one is detected at startup — no configuration required. The device in use is shown as a badge in the app header.

| Scenario | Device used | Speed |
|---|---|---|
| NVIDIA GPU (CUDA 12+) | `cuda` — `float16` | 5 – 20× faster than CPU |
| No GPU / Apple Silicon | `cpu` — `int8` | baseline |

> Apple Silicon MPS is **not yet supported** by the CTranslate2 backend. The app falls back to CPU on M-series Macs and still performs well (int8 quantised).

### Manual GPU setup (if the install script didn't detect your GPU)

```bash
# Activate the project venv first, then:
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

No CUDA toolkit installation is needed — the runtime libraries are shipped as Python wheels.

```
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
| Subtitle export + auto-save | ✅ Full support |
| Speaker diarization | ✅ Full support |
| AI dubbing (edge-tts) | ✅ Requires outbound HTTPS |
| Audio bitrate matching | ✅ Full support |
| Video bundling (MP4/MKV) | ✅ Full support |
| Media info (ffprobe) | ✅ Full support |
| Auto-open browser | ⚠️ May not work on headless servers |
| PyInstaller standalone build | ⚠️ Untested (likely works; not validated) |

---

## Repository

**https://github.com/alal76/subtitle-generator**
