"""
Video Subtitler — local Flask app
Transcribes video audio using Whisper and optionally translates subtitles
into multiple languages using argostranslate (fully offline).
"""

# Redirect HuggingFace + argos caches to a local directory.
# Works both when running from source and as a PyInstaller frozen bundle.
import os
import sys

if getattr(sys, "frozen", False):
    # Frozen (PyInstaller): place data next to the executable
    _PROJECT_DIR = os.path.dirname(sys.executable)
    _BUNDLE_DIR  = sys._MEIPASS  # where PyInstaller extracts bundled files
else:
    _PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    _BUNDLE_DIR  = _PROJECT_DIR

_LOCAL_CACHE = os.path.join(_PROJECT_DIR, ".local_cache")
os.makedirs(_LOCAL_CACHE, exist_ok=True)
os.environ["HF_HOME"]       = os.path.join(_LOCAL_CACHE, "huggingface")
os.environ["XDG_DATA_HOME"] = os.path.join(_LOCAL_CACHE, "data")


def _find_ffmpeg() -> str:
    """Return path to ffmpeg: bundled binary first, then system PATH."""
    candidates = [
        os.path.join(_BUNDLE_DIR, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return "ffmpeg"  # fall back to PATH


FFMPEG = _find_ffmpeg()

import asyncio
import json
import queue
import subprocess
import tempfile
import threading
import uuid
import webbrowser
from pathlib import Path

import argostranslate.package
import argostranslate.translate
from faster_whisper import WhisperModel
from flask import Flask, Response, jsonify, render_template_string, request, send_file

app = Flask(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".wmv"}
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

WHISPER_LANGUAGES = [
    ("auto", "Auto-detect"),
    ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("de", "German"),
    ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"),
    ("ru", "Russian"), ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ("ar", "Arabic"), ("tr", "Turkish"), ("sv", "Swedish"), ("da", "Danish"),
    ("fi", "Finnish"), ("no", "Norwegian"), ("he", "Hebrew"),
]

TRANSLATE_LANGUAGES = [
    ("en", "English"), ("es", "Spanish"), ("fr", "French"), ("de", "German"),
    ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch"), ("pl", "Polish"),
    ("ru", "Russian"), ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ("ar", "Arabic"), ("tr", "Turkish"), ("sv", "Swedish"), ("da", "Danish"),
    ("fi", "Finnish"), ("no", "Norwegian"), ("he", "Hebrew"),
]

LANG_NAME = dict(WHISPER_LANGUAGES)
LANG_NAME.update(dict(TRANSLATE_LANGUAGES))

# Default edge-tts neural voice for each supported language
EDGE_TTS_VOICES = {
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-EG-SalmaNeural",
    "tr": "tr-TR-EmelNeural",
    "sv": "sv-SE-SofieNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-NooraNeural",
    "no": "nb-NO-PernilleNeural",
    "he": "he-IL-HilaNeural",
}

_jobs: dict = {}
_dub_jobs: dict = {}
_pkg_index_fetched = False
_pkg_index_lock = threading.Lock()


# ── subtitle builders ─────────────────────────────────────────────────────────

def _fmt_srt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _fmt_vtt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def _build_srt(segs: list) -> str:
    lines = []
    for i, (start, end, text) in enumerate(segs, 1):
        lines += [str(i), f"{_fmt_srt(start)} --> {_fmt_srt(end)}", text, ""]
    return "\n".join(lines)


def _build_vtt(segs: list) -> str:
    lines = ["WEBVTT", ""]
    for start, end, text in segs:
        lines += [f"{_fmt_vtt(start)} --> {_fmt_vtt(end)}", text, ""]
    return "\n".join(lines)


# ── translation helpers ────────────────────────────────────────────────────────

def _ensure_pkg_index():
    global _pkg_index_fetched
    with _pkg_index_lock:
        if not _pkg_index_fetched:
            argostranslate.package.update_package_index()
            _pkg_index_fetched = True


def _ensure_translation(from_code: str, to_code: str):
    for lang in argostranslate.translate.get_installed_languages():
        if lang.code == from_code:
            for tl in lang.translations_to:
                if tl.code == to_code:
                    return
    _ensure_pkg_index()
    available = argostranslate.package.get_available_packages()
    pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
    if pkg is None:
        raise RuntimeError(f"No offline translation package available for {from_code} \u2192 {to_code}. "
                           "Try translating via English as an intermediate language.")
    argostranslate.package.install_from_path(pkg.download())


def _translate_segs(segs: list, from_code: str, to_code: str) -> list:
    _ensure_translation(from_code, to_code)
    result = []
    for start, end, text in segs:
        translated = argostranslate.translate.translate(text, from_code, to_code)
        result.append((start, end, translated))
    return result


# ── job worker ─────────────────────────────────────────────────────────────────

def _push(q, event: str, data: str):
    q.put(f"event: {event}\ndata: {data}\n\n")


def _run_job(job_id: str, video_path: str, model_size: str, language: str, translate_to: list):
    job = _jobs[job_id]
    q = job["queue"]
    audio_dir = os.path.join(_LOCAL_CACHE, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    audio_path = os.path.join(audio_dir, f"{job_id}.wav")
    try:
        _push(q, "progress", json.dumps({"msg": "Extracting audio\u2026", "pct": 5}))
        r = subprocess.run(
            [FFMPEG, "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {r.stderr.decode()}")

        _push(q, "progress", json.dumps({"msg": f"Loading Whisper \u2018{model_size}\u2019\u2026", "pct": 20}))
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        lang = None if language == "auto" else language
        _push(q, "progress", json.dumps({"msg": "Transcribing audio\u2026", "pct": 40}))
        segs_gen, info = model.transcribe(audio_path, language=lang, beam_size=5)
        raw_segments = list(segs_gen)

        detected = info.language
        segs = [(s.start, s.end, s.text.strip()) for s in raw_segments]

        # Keep audio for the dubbing feature
        job["audio_path"] = audio_path

        _push(q, "progress", json.dumps({"msg": "Building original subtitles\u2026", "pct": 75}))
        raw_segs_by_lang = {detected: segs}
        langs_result = {
            detected: {
                "plain": "\n".join(t for _, _, t in segs),
                "srt": _build_srt(segs),
                "vtt": _build_vtt(segs),
            }
        }

        total_trans = len([t for t in translate_to if t != detected])
        done_trans = 0
        for tgt in translate_to:
            if tgt == detected:
                continue
            pct = 75 + int(20 * (done_trans + 1) / max(total_trans, 1))
            name = LANG_NAME.get(tgt, tgt)
            _push(q, "progress", json.dumps({"msg": f"Translating to {name}\u2026", "pct": pct}))
            translated_segs = _translate_segs(segs, detected, tgt)
            raw_segs_by_lang[tgt] = translated_segs
            langs_result[tgt] = {
                "plain": "\n".join(t for _, _, t in translated_segs),
                "srt": _build_srt(translated_segs),
                "vtt": _build_vtt(translated_segs),
            }
            done_trans += 1

        job["result"] = {
            "detected_language": detected,
            "detected_language_name": LANG_NAME.get(detected, detected),
            "segment_count": len(segs),
            "langs": langs_result,
            "segments": raw_segs_by_lang,  # timed (start, end, text) per lang — used by dubbing
        }
        _push(q, "done", json.dumps({"msg": "Done!"}))
    except Exception as exc:
        job["error"] = str(exc)
        _push(q, "error", json.dumps({"msg": str(exc)}))
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
        # audio_path is intentionally kept for the dubbing feature
        q.put(None)


# ── dubbing helpers ────────────────────────────────────────────────────────────

async def _tts_generate(text: str, voice: str, output_path: str):
    """Generate one TTS segment and save as MP3 via edge-tts."""
    import edge_tts  # lazy import — only needed for dubbing
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def _dub_job(dub_id: str, job_id: str, lang: str):
    """
    Background thread that builds a dubbed audio track.

    Pipeline:
      1. Load the original 16-kHz WAV kept from transcription.
      2. Duck (–20 dB) the speech segments so background music stays audible.
      3. Generate TTS for every translated segment via edge-tts.
      4. Overlay each TTS clip at its original start timestamp.
      5. Export as 128 kbps MP3.
    """
    dub = _dub_jobs[dub_id]
    q = dub["queue"]
    try:
        job = _jobs.get(job_id)
        if not job or not job.get("result"):
            raise RuntimeError("Source transcription job not found or not complete.")

        audio_path = job.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError(
                "Original audio is no longer available. Re-run transcription to enable dubbing."
            )

        raw_segs = job["result"].get("segments", {}).get(lang)
        if not raw_segs:
            raise RuntimeError(f"No timed segments found for language '{lang}'.")

        voice = EDGE_TTS_VOICES.get(lang, "en-US-JennyNeural")

        _push(q, "progress", json.dumps({"msg": "Loading audio\u2026", "pct": 5}))
        from pydub import AudioSegment  # lazy import — only needed for dubbing

        orig_audio = AudioSegment.from_wav(audio_path)

        # Duck the original audio during each speech window so the background
        # score is preserved but the original voice is suppressed.
        _push(q, "progress", json.dumps({"msg": "Ducking original speech\u2026", "pct": 10}))
        ducked = orig_audio
        for start, end, _ in raw_segs:
            s_ms = int(start * 1000)
            e_ms = min(int(end * 1000), len(ducked))
            if s_ms >= e_ms:
                continue
            chunk = ducked[s_ms:e_ms] - 20  # −20 dB during speech
            ducked = ducked[:s_ms] + chunk + ducked[e_ms:]

        result_audio = ducked
        total = len(raw_segs)
        _push(q, "progress", json.dumps({"msg": f"Generating TTS (0/{total})\u2026", "pct": 15}))

        for i, (start, end, text) in enumerate(raw_segs):
            pct = 15 + int(75 * (i + 1) / max(total, 1))
            _push(q, "progress", json.dumps({
                "msg": f"Generating speech {i + 1}/{total}\u2026", "pct": pct
            }))
            if not text.strip():
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tts_path = tf.name
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_tts_generate(text, voice, tts_path))
                finally:
                    loop.close()
                tts_seg = AudioSegment.from_mp3(tts_path)
                result_audio = result_audio.overlay(tts_seg, position=int(start * 1000))
            finally:
                if os.path.exists(tts_path):
                    os.unlink(tts_path)

        _push(q, "progress", json.dumps({"msg": "Exporting MP3\u2026", "pct": 93}))
        out_dir = os.path.join(_LOCAL_CACHE, "audio")
        out_path = os.path.join(out_dir, f"{dub_id}_{lang}.mp3")
        result_audio.export(out_path, format="mp3", bitrate="128k")

        dub["result_path"] = out_path
        _push(q, "done", json.dumps({"msg": "Dubbed audio ready!"}))
    except Exception as exc:
        dub["error"] = str(exc)
        _push(q, "error", json.dumps({"msg": str(exc)}))
    finally:
        q.put(None)


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(
        HTML_UI,
        models=MODEL_SIZES,
        whisper_languages=WHISPER_LANGUAGES,
        translate_languages=TRANSLATE_LANGUAGES,
    )


@app.route("/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No file provided"), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return jsonify(error=f"Unsupported format '{ext}'"), 400
    model_size = request.form.get("model", "base")
    language = request.form.get("language", "auto")
    translate_to = request.form.getlist("translate_to")

    valid_trans = {c for c, _ in TRANSLATE_LANGUAGES}
    translate_to = [t for t in translate_to if t in valid_trans]

    if model_size not in MODEL_SIZES:
        return jsonify(error="Invalid model"), 400
    if language not in dict(WHISPER_LANGUAGES):
        return jsonify(error="Invalid language"), 400

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    f.save(tmp.name)
    tmp.close()

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"queue": queue.Queue(), "result": None, "error": None}
    threading.Thread(
        target=_run_job,
        args=(job_id, tmp.name, model_size, language, translate_to),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id)


@app.route("/stream/<job_id>")
def stream(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404

    def generate():
        q = job["queue"]
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/result/<job_id>")
def result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    if job["error"]:
        return jsonify(error=job["error"]), 500
    if not job["result"]:
        return jsonify(error="Not ready yet"), 202
    return jsonify(job["result"])


@app.route("/download/<job_id>/<lang>/<fmt>")
def download(job_id: str, lang: str, fmt: str):
    job = _jobs.get(job_id)
    if not job or not job["result"]:
        return jsonify(error="Not found"), 404
    if fmt not in ("srt", "vtt", "txt"):
        return jsonify(error="Invalid format"), 400
    langs = job["result"].get("langs", {})
    if lang not in langs:
        return jsonify(error="Language not available"), 404
    key = "plain" if fmt == "txt" else fmt
    content = langs[lang][key]
    tmp = tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False, mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return send_file(tmp.name, as_attachment=True, download_name=f"subtitles_{lang}.{fmt}")


@app.route("/dub/<job_id>/<lang>", methods=["POST"])
def start_dub(job_id: str, lang: str):
    job = _jobs.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="Transcription job not found"), 404
    if lang not in job["result"].get("langs", {}):
        return jsonify(error="Language not available for this job"), 400
    dub_id = str(uuid.uuid4())
    _dub_jobs[dub_id] = {"queue": queue.Queue(), "result_path": None, "error": None}
    threading.Thread(
        target=_dub_job,
        args=(dub_id, job_id, lang),
        daemon=True,
    ).start()
    return jsonify(dub_id=dub_id)


@app.route("/stream_dub/<dub_id>")
def stream_dub(dub_id: str):
    dub = _dub_jobs.get(dub_id)
    if not dub:
        return jsonify(error="Unknown dub job"), 404

    def generate():
        q = dub["queue"]
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/download_dub/<dub_id>")
def download_dub(dub_id: str):
    dub = _dub_jobs.get(dub_id)
    if not dub:
        return jsonify(error="Not found"), 404
    if dub["error"]:
        return jsonify(error=dub["error"]), 500
    path = dub.get("result_path")
    if not path or not os.path.exists(path):
        return jsonify(error="Audio not ready"), 202
    return send_file(path, as_attachment=True, download_name="dubbed_audio.mp3")


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Video Subtitler</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
header{background:#1e293b;padding:1.25rem 2rem;border-bottom:1px solid #334155;display:flex;align-items:center;gap:.75rem}
header h1{font-size:1.35rem;font-weight:700}
.badge{background:#6366f1;color:#fff;font-size:.7rem;padding:2px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.05em}
main{max-width:960px;margin:2rem auto;padding:0 1rem}
.card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem}
.card h2{font-size:1rem;font-weight:600;margin-bottom:1rem;color:#94a3b8}
.drop-zone{border:2px dashed #475569;border-radius:8px;padding:3rem 1rem;text-align:center;cursor:pointer;transition:border-color .2s,background .2s}
.drop-zone:hover,.drop-zone.dragover{border-color:#6366f1;background:#1e2a45}
.drop-zone svg{opacity:.4;margin-bottom:.75rem}
.drop-zone p{color:#94a3b8;font-size:.9rem}
.drop-zone strong{color:#c7d2fe}
#file-name{margin-top:.5rem;font-size:.85rem;color:#818cf8}
.row{display:flex;gap:1rem;flex-wrap:wrap}
.field{display:flex;flex-direction:column;gap:.35rem;flex:1;min-width:160px}
label{font-size:.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}
select{background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:6px;padding:.5rem .75rem;font-size:.9rem}
.lang-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:.4rem;margin-top:.5rem}
.lang-check{display:flex;align-items:center;gap:.4rem;background:#0f172a;border:1px solid #334155;border-radius:6px;padding:.35rem .6rem;cursor:pointer;transition:border-color .15s}
.lang-check:hover{border-color:#6366f1}
.lang-check input{accent-color:#6366f1;cursor:pointer}
.lang-check span{font-size:.82rem;color:#cbd5e1}
.btn{background:#6366f1;color:#fff;border:none;border-radius:8px;padding:.65rem 1.5rem;font-size:.95rem;font-weight:600;cursor:pointer;transition:background .2s}
.btn:hover:not(:disabled){background:#4f46e5}
.btn:disabled{opacity:.5;cursor:not-allowed}
.progress-wrap{display:none}
.progress-bar-bg{background:#334155;border-radius:999px;height:8px;margin:.5rem 0}
.progress-bar{background:#6366f1;height:8px;border-radius:999px;width:0%;transition:width .4s}
.progress-msg{font-size:.85rem;color:#94a3b8}
#results-card{display:none}
.meta{font-size:.85rem;color:#64748b;margin-bottom:1rem}
.meta span{color:#818cf8;font-weight:600}
.lang-tabs{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:1rem}
.ltab{background:transparent;border:1px solid #334155;color:#94a3b8;border-radius:6px;padding:.35rem .9rem;font-size:.85rem;cursor:pointer;transition:all .15s}
.ltab.active{background:#6366f1;border-color:#6366f1;color:#fff}
.fmt-bar{display:flex;gap:.4rem;margin-bottom:.6rem}
.ftab{background:transparent;border:1px solid #334155;color:#94a3b8;border-radius:6px;padding:.25rem .7rem;font-size:.8rem;cursor:pointer;transition:all .15s}
.ftab.active{background:#334155;color:#e2e8f0}
textarea{width:100%;background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:8px;padding:.75rem;font-size:.85rem;font-family:monospace;resize:vertical;line-height:1.55}
.dl-bar{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.75rem}
.dl-btn{background:#0f4c75;color:#7dd3fc;border:1px solid #0369a1;border-radius:6px;padding:.4rem 1rem;font-size:.82rem;text-decoration:none;display:inline-block;transition:background .15s}
.dl-btn:hover{background:#0369a1}
.error-box{background:#450a0a;border:1px solid #7f1d1d;color:#fca5a5;border-radius:8px;padding:.75rem 1rem;font-size:.9rem;display:none;margin-top:.75rem}
.dub-note{font-size:.82rem;color:#64748b;line-height:1.6;margin-bottom:.75rem}
.success-msg{color:#4ade80;font-size:.9rem;margin-bottom:.5rem}
</style>
</head>
<body>
<header>
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2">
    <rect x="2" y="2" width="20" height="20" rx="3"/>
    <polygon points="10,8 16,12 10,16" fill="#818cf8" stroke="none"/>
  </svg>
  <h1>Video Subtitler</h1>
  <span class="badge">local</span>
</header>

<main>
  <!-- Step 1 -->
  <div class="card">
    <h2>1 — Upload video</h2>
    <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M12 4v12M8 8l4-4 4 4"/>
      </svg>
      <p><strong>Click to browse</strong> or drag &amp; drop</p>
      <p style="margin-top:.3rem">MP4 &middot; MKV &middot; AVI &middot; MOV &middot; WEBM &middot; FLV &middot; M4V &middot; WMV</p>
      <div id="file-name"></div>
    </div>
    <input type="file" id="file-input" accept=".mp4,.mkv,.avi,.mov,.webm,.flv,.m4v,.wmv" style="display:none"/>
  </div>

  <!-- Step 2 -->
  <div class="card">
    <h2>2 — Transcription options</h2>
    <div class="row">
      <div class="field">
        <label for="model-sel">Whisper model</label>
        <select id="model-sel">
          {% for m in models %}
          <option value="{{ m }}"{% if m == "base" %} selected{% endif %}>{{ m }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="field">
        <label for="lang-sel">Audio language</label>
        <select id="lang-sel">
          {% for code, name in whisper_languages %}
          <option value="{{ code }}">{{ name }}</option>
          {% endfor %}
        </select>
      </div>
    </div>
    <p style="margin-top:.6rem;font-size:.8rem;color:#475569">tiny/base = fast &middot; small/medium = accurate &middot; large-v3 = best</p>
  </div>

  <!-- Step 3 -->
  <div class="card">
    <h2>3 — Translate subtitles to <span style="color:#94a3b8;font-weight:400">(optional — select one or more)</span></h2>
    <div class="lang-grid" id="trans-grid">
      {% for code, name in translate_languages %}
      <label class="lang-check">
        <input type="checkbox" name="translate_to" value="{{ code }}"/>
        <span>{{ name }}</span>
      </label>
      {% endfor %}
    </div>
    <p style="margin-top:.6rem;font-size:.8rem;color:#475569">
      Language packages are downloaded once on first use (~50 MB each) and cached locally.
    </p>
  </div>

  <!-- Step 4 -->
  <div class="card">
    <h2>4 — Generate</h2>
    <button class="btn" id="run-btn" disabled>Generate subtitles</button>
    <div class="progress-wrap" id="progress-wrap" style="margin-top:1rem">
      <div class="progress-bar-bg"><div class="progress-bar" id="progress-bar"></div></div>
      <div class="progress-msg" id="progress-msg">Starting&hellip;</div>
    </div>
    <div class="error-box" id="error-box"></div>
  </div>

  <!-- Results -->
  <div class="card" id="results-card">
    <h2>5 — Results</h2>
    <div class="meta" id="meta"></div>
    <div class="lang-tabs" id="lang-tabs"></div>
    <div class="fmt-bar" id="fmt-bar">
      <button class="ftab active" data-fmt="plain">Transcript</button>
      <button class="ftab" data-fmt="srt">SRT</button>
      <button class="ftab" data-fmt="vtt">VTT</button>
    </div>
    <textarea id="text-out" rows="16" readonly></textarea>
    <div class="dl-bar" id="dl-bar"></div>
  </div>

  <!-- Step 6: Dub Audio -->
  <div class="card" id="dub-card" style="display:none">
    <h2>6 &mdash; Dub Audio <span style="color:#94a3b8;font-weight:400">(replaces dialogue with AI voice)</span></h2>
    <p class="dub-note">Generates a new audio track in the selected language: background music and sound effects are preserved (briefly ducked during speech), and all dialogue is replaced with neural text-to-speech. Internet access required for voice synthesis.</p>
    <div class="row" style="align-items:flex-end">
      <div class="field">
        <label for="dub-lang-sel">Target language</label>
        <select id="dub-lang-sel"></select>
      </div>
      <button class="btn" id="dub-btn" style="margin-bottom:1px">Generate dubbed audio</button>
    </div>
    <div id="dub-prog-wrap" style="display:none;margin-top:1rem">
      <div class="progress-bar-bg"><div class="progress-bar" id="dub-prog-bar"></div></div>
      <div class="progress-msg" id="dub-prog-msg">Starting&hellip;</div>
    </div>
    <div class="error-box" id="dub-error-box"></div>
    <div id="dub-result-area" style="display:none;margin-top:1rem">
      <p class="success-msg">&checkmark; Dubbed audio ready</p>
      <a id="dub-dl-btn" class="dl-btn" href="#">&#11015; Download dubbed audio (.mp3)</a>
    </div>
  </div>
</main>

<script>
let selectedFile = null, currentLang = null, currentFmt = "plain";
let resultData = null, jobId = null;

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const fileName  = document.getElementById("file-name");
const runBtn    = document.getElementById("run-btn");
const progWrap  = document.getElementById("progress-wrap");
const progBar   = document.getElementById("progress-bar");
const progMsg   = document.getElementById("progress-msg");
const errorBox  = document.getElementById("error-box");
const resultsCard = document.getElementById("results-card");
const textOut   = document.getElementById("text-out");
const metaEl    = document.getElementById("meta");
const langTabs  = document.getElementById("lang-tabs");
const dlBar     = document.getElementById("dl-bar");

function setFile(f) {
  selectedFile = f;
  fileName.textContent = f ? f.name : "";
  runBtn.disabled = !f;
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

document.querySelectorAll(".ftab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".ftab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    currentFmt = btn.dataset.fmt;
    renderText();
  });
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  errorBox.style.display = "none";
  resultsCard.style.display = "none";
  progWrap.style.display = "block";
  progBar.style.width = "0%";
  progMsg.textContent = "Uploading\u2026";
  runBtn.disabled = true;

  const fd = new FormData();
  fd.append("video", selectedFile);
  fd.append("model", document.getElementById("model-sel").value);
  fd.append("language", document.getElementById("lang-sel").value);
  document.querySelectorAll('input[name="translate_to"]:checked').forEach(cb => {
    fd.append("translate_to", cb.value);
  });

  let resp;
  try { resp = await fetch("/transcribe", { method: "POST", body: fd }); }
  catch (e) { showError("Upload failed: " + e.message); return; }
  if (!resp.ok) { const j = await resp.json().catch(()=>({})); showError(j.error||"Upload failed"); return; }

  const data = await resp.json();
  jobId = data.job_id;
  listenProgress(jobId);
});

function listenProgress(jid) {
  const es = new EventSource("/stream/" + jid);
  es.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    progMsg.textContent = d.msg;
    progBar.style.width = (d.pct || 0) + "%";
  });
  es.addEventListener("done", async () => {
    es.close();
    progBar.style.width = "100%";
    progMsg.textContent = "Done!";
    const r = await fetch("/result/" + jid);
    const data = await r.json();
    showResults(jid, data);
  });
  es.addEventListener("error", e => {
    es.close();
    try { showError(JSON.parse(e.data).msg); } catch(_) { showError("Transcription failed."); }
  });
}

function showResults(jid, data) {
  resultData = data;
  jobId = jid;
  metaEl.innerHTML =
    "Detected language: <span>" + (data.detected_language_name || data.detected_language) + "</span>" +
    " &nbsp;|&nbsp; Segments: <span>" + data.segment_count + "</span>" +
    " &nbsp;|&nbsp; Versions: <span>" + Object.keys(data.langs).length + "</span>";

  // Build language tabs
  langTabs.innerHTML = "";
  const langNames = {
    en:"English",es:"Spanish",fr:"French",de:"German",it:"Italian",pt:"Portuguese",
    nl:"Dutch",pl:"Polish",ru:"Russian",ja:"Japanese",ko:"Korean",zh:"Chinese",
    ar:"Arabic",tr:"Turkish",sv:"Swedish",da:"Danish",fi:"Finnish",no:"Norwegian",he:"Hebrew"
  };
  let first = true;
  for (const [code, _] of Object.entries(data.langs)) {
    const btn = document.createElement("button");
    btn.className = "ltab" + (first ? " active" : "");
    btn.dataset.lang = code;
    const isDetected = (code === data.detected_language);
    btn.textContent = (langNames[code] || code) + (isDetected ? " (original)" : "");
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ltab").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      currentLang = code;
      renderText();
      renderDownloads();
    });
    langTabs.appendChild(btn);
    if (first) { currentLang = code; first = false; }
  }

  renderText();
  renderDownloads();
  initDubCard(data);
  resultsCard.style.display = "block";
  runBtn.disabled = false;
}

function renderText() {
  if (!resultData || !currentLang) return;
  const langData = resultData.langs[currentLang];
  if (!langData) return;
  const key = currentFmt === "plain" ? "plain" : currentFmt;
  textOut.value = langData[key] || "";
}

function renderDownloads() {
  if (!resultData || !currentLang) return;
  const langNames = {
    en:"English",es:"Spanish",fr:"French",de:"German",it:"Italian",pt:"Portuguese",
    nl:"Dutch",pl:"Polish",ru:"Russian",ja:"Japanese",ko:"Korean",zh:"Chinese",
    ar:"Arabic",tr:"Turkish",sv:"Swedish",da:"Danish",fi:"Finnish",no:"Norwegian",he:"Hebrew"
  };
  const name = langNames[currentLang] || currentLang;
  dlBar.innerHTML = "";
  for (const fmt of ["srt","vtt","txt"]) {
    const a = document.createElement("a");
    a.className = "dl-btn";
    a.href = `/download/${jobId}/${currentLang}/${fmt}`;
    a.textContent = `\u2b07 ${name} .${fmt}`;
    dlBar.appendChild(a);
  }
}

// ── Dub Audio ─────────────────────────────────────────────────────────────────
const dubCard       = document.getElementById("dub-card");
const dubLangSel    = document.getElementById("dub-lang-sel");
const dubBtn        = document.getElementById("dub-btn");
const dubProgWrap   = document.getElementById("dub-prog-wrap");
const dubProgBar    = document.getElementById("dub-prog-bar");
const dubProgMsg    = document.getElementById("dub-prog-msg");
const dubErrorBox   = document.getElementById("dub-error-box");
const dubResultArea = document.getElementById("dub-result-area");
const dubDlBtn      = document.getElementById("dub-dl-btn");
const dubLangNames  = {
  en:"English",es:"Spanish",fr:"French",de:"German",it:"Italian",pt:"Portuguese",
  nl:"Dutch",pl:"Polish",ru:"Russian",ja:"Japanese",ko:"Korean",zh:"Chinese",
  ar:"Arabic",tr:"Turkish",sv:"Swedish",da:"Danish",fi:"Finnish",no:"Norwegian",he:"Hebrew"
};

function initDubCard(data) {
  dubLangSel.innerHTML = "";
  for (const code of Object.keys(data.langs)) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = (dubLangNames[code] || code) +
                      (code === data.detected_language ? " (original)" : "");
    dubLangSel.appendChild(opt);
  }
  dubProgWrap.style.display  = "none";
  dubErrorBox.style.display  = "none";
  dubResultArea.style.display = "none";
  dubBtn.disabled = false;
  dubCard.style.display = "block";
}

dubBtn.addEventListener("click", async () => {
  const lang = dubLangSel.value;
  if (!jobId || !lang) return;
  dubErrorBox.style.display  = "none";
  dubResultArea.style.display = "none";
  dubProgWrap.style.display  = "block";
  dubProgBar.style.width = "0%";
  dubProgMsg.textContent = "Starting\u2026";
  dubBtn.disabled = true;

  let resp;
  try { resp = await fetch(`/dub/${jobId}/${lang}`, { method: "POST" }); }
  catch (e) { showDubError("Request failed: " + e.message); return; }
  if (!resp.ok) {
    const j = await resp.json().catch(() => ({}));
    showDubError(j.error || "Failed to start dub job");
    return;
  }
  const { dub_id } = await resp.json();

  const es = new EventSource(`/stream_dub/${dub_id}`);
  es.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    dubProgMsg.textContent = d.msg;
    dubProgBar.style.width = (d.pct || 0) + "%";
  });
  es.addEventListener("done", () => {
    es.close();
    dubProgBar.style.width = "100%";
    dubProgMsg.textContent = "Done!";
    const name = dubLangNames[lang] || lang;
    dubDlBtn.textContent = `\u2b07 Download dubbed audio \u2014 ${name} (.mp3)`;
    dubDlBtn.href = `/download_dub/${dub_id}`;
    dubResultArea.style.display = "block";
    dubBtn.disabled = false;
  });
  es.addEventListener("error", e => {
    es.close();
    try { showDubError(JSON.parse(e.data).msg); } catch(_) { showDubError("Dubbing failed."); }
  });
});

function showDubError(msg) {
  dubProgWrap.style.display = "none";
  dubErrorBox.textContent = msg;
  dubErrorBox.style.display = "block";
  dubBtn.disabled = false;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, threaded=True, debug=False)
