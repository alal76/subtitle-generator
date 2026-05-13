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
    _BUNDLE_DIR  = getattr(sys, "_MEIPASS", _PROJECT_DIR)
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


def _find_ffprobe() -> str:
    """Return path to ffprobe: same directory as ffmpeg first, then PATH."""
    ffmpeg_dir = os.path.dirname(os.path.abspath(FFMPEG))
    name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"
    candidate = os.path.join(ffmpeg_dir, name)
    if os.path.isfile(candidate):
        return candidate
    return "ffprobe"  # fall back to PATH


FFPROBE = _find_ffprobe()

import asyncio
import io
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
_mux_jobs: dict = {}
_config: dict = {"output_folder": ""}
_pkg_index_fetched = False
_pkg_index_lock = threading.Lock()

# ── device detection (runs once at startup) ───────────────────────────────────
def _detect_device() -> tuple[str, str]:
    """Return (device, compute_type) for transcription.

    Priority: CUDA (NVIDIA) > Apple Silicon MLX > CPU
    """
    # 1. NVIDIA CUDA via CTranslate2 (faster-whisper)
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    # 2. Apple Silicon GPU/ANE via MLX
    if sys.platform == "darwin":
        try:
            import platform as _plat
            if _plat.machine() == "arm64":
                import mlx_whisper  # noqa: F401 — just probing for availability
                return "mlx", "float16"
        except Exception:
            pass
    # 3. CPU fallback (int8 via faster-whisper)
    return "cpu", "int8"

_DEVICE, _COMPUTE_TYPE = _detect_device()
_DEVICE_LABEL = {
    "cuda": "GPU (CUDA)",
    "mlx":  "Apple Silicon (MLX)",
    "cpu":  "CPU",
}.get(_DEVICE, "CPU")

# MLX repo mapping: faster-whisper uses size names; mlx-whisper needs HF repo ids.
_MLX_REPO = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# Response-literal constants (suppress duplicate-string warnings)
_ERR_UNKNOWN_JOB = "Unknown job"
_ERR_NOT_FOUND = "Not found"
_MIME_SSE = "text/event-stream"
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


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
                if tl.to_lang.code == to_code:
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


# ── media helpers ─────────────────────────────────────────────────────────────

def _run_ffprobe(file_path: str) -> dict:
    """Return ffprobe JSON for a video/audio file."""
    r = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", file_path],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe error: {r.stderr.decode()}")
    return json.loads(r.stdout.decode())


def _transcribe(audio_path: str, model_size: str, language):
    """Run Whisper on `audio_path` using the best available backend.

    Returns (segs, detected_lang) where segs = [(start, end, text), ...]
    and detected_lang is a 2-letter ISO code.
    """
    if _DEVICE == "mlx":
        # Apple Silicon GPU / Neural Engine path
        import mlx_whisper  # type: ignore[import-not-found]
        repo = _MLX_REPO.get(model_size, _MLX_REPO["small"])
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=repo,
            language=language,
            word_timestamps=False,
        )
        segs = [(float(s["start"]), float(s["end"]), s["text"].strip())
                for s in result.get("segments", [])]
        return segs, result.get("language") or (language or "en")
    # Default: faster-whisper (CPU int8 or CUDA float16)
    model = WhisperModel(model_size, device=_DEVICE, compute_type=_COMPUTE_TYPE)
    segs_gen, info = model.transcribe(audio_path, language=language, beam_size=5)
    segs = [(s.start, s.end, s.text.strip()) for s in segs_gen]
    return segs, info.language


# ── speaker diarization ───────────────────────────────────────────────────────

def _diarize(audio_path: str, segs: list) -> tuple:
    """
    Cluster Whisper segments into speaker identities using MFCC features.

    Returns:
        speakers  – {speaker_id: {gender_hint, segment_count, total_duration, median_f0}}
        spk_segs  – [(start, end, text, speaker_id), …]

    Falls back to single speaker (SPEAKER_00) if librosa/sklearn are unavailable.
    """
    try:
        import librosa
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        spk_segs = [(s, e, t, "SPEAKER_00") for s, e, t in segs]
        total_dur = sum(e - s for s, e, t in segs)
        return (
            {"SPEAKER_00": {"gender_hint": "U", "segment_count": len(segs),
                            "total_duration": round(total_dur, 2), "median_f0": 0.0,
                            "speaking_wps": 0.0, "suggested_rate": 0, "suggested_pitch": 0}},
            spk_segs,
        )

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    features, f0_per_seg = [], []

    for start, end, _ in segs:
        s = int(start * sr)
        e = min(int(end * sr), len(y))
        chunk = y[s:e] if e > s else np.zeros(2048)
        if len(chunk) < 2048:
            chunk = np.pad(chunk, (0, 2048 - len(chunk)))
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        features.append(np.mean(mfcc, axis=1))
        f0 = librosa.yin(chunk, fmin=60, fmax=500, sr=sr)
        valid = f0[f0 > 0]
        f0_per_seg.append(float(np.median(valid)) if len(valid) else 160.0)

    if len(features) < 2:
        gender = "F" if f0_per_seg[0] > 165 else "M"
        spk_segs = [(segs[0][0], segs[0][1], segs[0][2], "SPEAKER_00")]
        dur = segs[0][1] - segs[0][0]
        words = len(segs[0][2].split())
        wps = words / dur if dur > 0 else 2.5
        f0_bl = 200.0 if gender == "F" else 120.0
        return (
            {"SPEAKER_00": {"gender_hint": gender, "segment_count": 1,
                            "total_duration": round(dur, 2),
                            "median_f0": round(f0_per_seg[0], 1),
                            "speaking_wps": round(wps, 2),
                            "suggested_rate": 0,
                            "suggested_pitch": int(max(-10, min(10, round((f0_per_seg[0] - f0_bl) / 10) * 2)))}},
            spk_segs,
        )

    X = StandardScaler().fit_transform(np.array(features))
    # Auto-detect optimal speaker count via silhouette scoring — no hard cap
    from sklearn.metrics import silhouette_score as _sil
    n = len(features)
    max_k = max(2, min(12, n // 3 + 1))
    if n > 2:
        max_k = min(max_k, n - 1)
    best_k, best_score = 2, -1.0
    for k in range(2, max_k + 1):
        lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        sc = _sil(X, lbl) if k < n else 0.0
        if sc > best_score:
            best_k, best_score = k, sc
    labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    return _build_speaker_dicts(segs, labels, f0_per_seg)


def _build_speaker_dicts(segs: list, labels, f0_per_seg: list) -> tuple:
    """Convert cluster labels into speaker dicts and labelled segment list."""
    import numpy as np

    raw: dict = {}
    for i, (start, end, text) in enumerate(segs):
        spk = f"SPEAKER_{labels[i]:02d}"
        raw.setdefault(spk, {"idxs": [], "f0": []})
        raw[spk]["idxs"].append(i)
        raw[spk]["f0"].append(f0_per_seg[i])

    _BASELINE_WPS = 2.5  # typical speaking rate (words per second)
    speakers: dict = {}
    spk_segs_list = [()] * len(segs)
    for spk, data in sorted(raw.items()):
        med_f0 = float(np.median(data["f0"]))
        gender = "F" if med_f0 > 165 else "M"
        total_dur = sum(segs[i][1] - segs[i][0] for i in data["idxs"])
        # Estimate speaking rate and derive TTS rate/pitch offsets
        total_words = sum(len(segs[i][2].split()) for i in data["idxs"])
        wps = total_words / total_dur if total_dur > 0 else _BASELINE_WPS
        rate_frac = (wps - _BASELINE_WPS) / _BASELINE_WPS
        suggested_rate = int(max(-20, min(20, round(rate_frac * 50 / 2) * 2)))
        f0_baseline = 200.0 if gender == "F" else 120.0
        suggested_pitch = int(max(-10, min(10, round((med_f0 - f0_baseline) / 10) * 2)))
        speakers[spk] = {
            "gender_hint": gender,
            "segment_count": len(data["idxs"]),
            "total_duration": round(total_dur, 2),
            "median_f0": round(med_f0, 1),
            "speaking_wps": round(wps, 2),
            "suggested_rate": suggested_rate,
            "suggested_pitch": suggested_pitch,
        }
        for i in data["idxs"]:
            spk_segs_list[i] = (segs[i][0], segs[i][1], segs[i][2], spk)

    return speakers, spk_segs_list


# ── mux helper ────────────────────────────────────────────────────────────────

def _mux_job(mux_id: str, video_path: str, audio_path: str,
             include_original: bool, include_dubbed: bool,
             out_ext: str, audio_bitrate: str = "192k",
             job_id: str = ""):
    """
    Background thread: bundle video with selected audio tracks.
    include_original – keep all original audio tracks from the source video
    include_dubbed   – add the TTS-dubbed audio track
    out_ext          – '.mp4' or '.mkv'
    audio_bitrate    – AAC target bitrate, matched to source (e.g. '192k')
    """
    mux = _mux_jobs[mux_id]
    q = mux["queue"]
    out_dir = os.path.join(_LOCAL_CACHE, "mux")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{mux_id}{out_ext}")
    try:
        _push(q, "progress", json.dumps({"msg": "Bundling tracks…", "pct": 10}))
        # Build ffmpeg inputs and stream maps
        inputs = [FFMPEG, "-y", "-i", video_path]
        maps = ["-map", "0:v:0"]
        dubbed_idx = 1
        if include_dubbed:
            inputs += ["-i", audio_path]
        if include_original:
            maps += ["-map", "0:a?"]
        if include_dubbed:
            maps += ["-map", f"{dubbed_idx}:a:0"]
        # Transcode to AAC when dubbed track is present; copy otherwise
        if include_dubbed:
            audio_args = ["-c:a", "aac", "-b:a", audio_bitrate]
        else:
            audio_args = ["-c:a", "copy"]
        cmd = inputs + maps + ["-c:v", "copy"] + audio_args + [out_path]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {r.stderr.decode()}")
        mux["result_path"] = out_path
        # Auto-save bundled video to configured output folder
        _out_folder = _config.get("output_folder", "")
        if _out_folder:
            try:
                import shutil as _shutil
                os.makedirs(_out_folder, exist_ok=True)
                _src_job = _jobs.get(job_id, {}) if job_id else {}
                _base = Path(_src_job.get("original_filename", f"job_{mux_id[:8]}")).stem
                _shutil.copy2(out_path, os.path.join(_out_folder, f"{_base}_bundled{out_ext}"))
            except Exception as _e:
                app.logger.warning("Auto-save bundled video failed: %s", _e)
        _push(q, "done", json.dumps({"msg": "Video ready!"}))
    except Exception as exc:
        mux["error"] = str(exc)
        _push(q, "error", json.dumps({"msg": str(exc)}))
    finally:
        q.put(None)


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

        # Detect source audio bitrate so the dubbed MP3 can match it
        try:
            _probe = _run_ffprobe(video_path)
            for _s in _probe.get("streams", []):
                if _s.get("codec_type") == "audio" and _s.get("bit_rate"):
                    _kb = max(64, min(320, int(int(_s["bit_rate"]) / 1000)))
                    job["source_audio_bitrate"] = f"{_kb}k"
                    break
            else:
                job["source_audio_bitrate"] = "192k"
        except Exception:
            job["source_audio_bitrate"] = "192k"

        _push(q, "progress", json.dumps({"msg": f"Loading Whisper \u2018{model_size}\u2019 on {_DEVICE_LABEL}\u2026", "pct": 20}))
        lang = None if language == "auto" else language
        _push(q, "progress", json.dumps({"msg": "Transcribing audio\u2026", "pct": 40}))
        segs, detected = _transcribe(audio_path, model_size, lang)

        # Keep audio and video for dubbing / muxing features
        job["audio_path"] = audio_path
        job["video_path"] = video_path  # persisted — not deleted on success

        _push(q, "progress", json.dumps({"msg": "Detecting speakers\u2026", "pct": 60}))
        try:
            speakers, speaker_segs = _diarize(audio_path, segs)
        except Exception:
            speaker_segs = [(s, e, t, "SPEAKER_00") for s, e, t in segs]
            speakers = {"SPEAKER_00": {"gender_hint": "U", "segment_count": len(segs),
                                       "total_duration": sum(e - s for s, e, t in segs),
                                       "median_f0": 0.0, "speaking_wps": 0.0,
                                       "suggested_rate": 0, "suggested_pitch": 0}}
        job["speakers"] = speakers
        job["speaker_segs"] = speaker_segs

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
        # Auto-save subtitle files to configured output folder
        _out_folder = _config.get("output_folder", "")
        if _out_folder:
            try:
                os.makedirs(_out_folder, exist_ok=True)
                _base = Path(job.get("original_filename", f"job_{job_id[:8]}")).stem
                for _lang, _ldata in langs_result.items():
                    for _fmt, _ext in [("srt", "srt"), ("vtt", "vtt"), ("plain", "txt")]:
                        _content = _ldata.get(_fmt, "")
                        if _content:
                            with open(os.path.join(_out_folder, f"{_base}_{_lang}.{_ext}"), "w", encoding="utf-8") as _f:
                                _f.write(_content)
            except Exception as _e:
                app.logger.warning("Auto-save subtitles failed: %s", _e)
        _push(q, "done", json.dumps({"msg": "Done!"}))
    except Exception as exc:
        job["error"] = str(exc)
        _push(q, "error", json.dumps({"msg": str(exc)}))
        # Clean up temp files on failure
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        if os.path.exists(video_path):
            os.unlink(video_path)
    finally:
        # audio_path and video_path kept on success for dubbing/muxing
        q.put(None)


# ── dubbing helpers ────────────────────────────────────────────────────────────

async def _tts_generate(text: str, voice: str, output_path: str,
                        rate: str = "+0%", pitch: str = "+0Hz",
                        retries: int = 3):
    """Generate one TTS segment and save as MP3 via edge-tts.

    Retries up to `retries` times on NoAudioReceived before re-raising.
    """
    import asyncio as _asyncio
    import edge_tts  # lazy import — only needed for dubbing
    last_exc: Exception = RuntimeError("TTS failed after retries")
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await communicate.save(output_path)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                await _asyncio.sleep(1.5 * (attempt + 1))
    raise last_exc


def _dub_job(dub_id: str, job_id: str, lang: str, voice_profiles=None):
    """
    Background thread that builds a dubbed audio track.

    voice_profiles: {speaker_id: {"voice": "…", "rate": "+2%", "pitch": "+5Hz"}}
    Falls back to EDGE_TTS_VOICES[lang] for any speaker not in voice_profiles.
    """
    dub = _dub_jobs[dub_id]
    q = dub["queue"]
    voice_profiles = voice_profiles or {}
    default_voice = EDGE_TTS_VOICES.get(lang, "en-US-JennyNeural")
    try:
        audio_path, raw_segs, seg_to_speaker = _prepare_dub_inputs(job_id, lang)

        _push(q, "progress", json.dumps({"msg": "Loading audio\u2026", "pct": 5}))
        from pydub import AudioSegment  # lazy import — only needed for dubbing
        # Tell pydub where our ffmpeg binary is (critical on Windows when ffmpeg
        # is bundled and not on the system PATH).
        AudioSegment.converter = FFMPEG

        orig_audio = AudioSegment.from_wav(audio_path)

        _push(q, "progress", json.dumps({"msg": "Ducking original speech\u2026", "pct": 10}))
        ducked = _duck_speech(orig_audio, raw_segs)

        result_audio = _overlay_tts(
            ducked, raw_segs, seg_to_speaker, voice_profiles, default_voice, q
        )

        _push(q, "progress", json.dumps({"msg": "Exporting MP3\u2026", "pct": 93}))
        out_dir = os.path.join(_LOCAL_CACHE, "audio")
        out_path = os.path.join(out_dir, f"{dub_id}_{lang}.mp3")
        src_bitrate = _jobs.get(job_id, {}).get("source_audio_bitrate", "192k")
        result_audio.export(out_path, format="mp3", bitrate=src_bitrate)

        dub["result_path"] = out_path
        # Auto-save dubbed audio to configured output folder
        _out_folder = _config.get("output_folder", "")
        if _out_folder:
            try:
                import shutil as _shutil
                os.makedirs(_out_folder, exist_ok=True)
                _src_job = _jobs.get(job_id, {})
                _base = Path(_src_job.get("original_filename", f"job_{job_id[:8]}")).stem
                _shutil.copy2(out_path, os.path.join(_out_folder, f"{_base}_{lang}_dubbed.mp3"))
            except Exception as _e:
                app.logger.warning("Auto-save dubbed audio failed: %s", _e)
        _push(q, "done", json.dumps({"msg": "Dubbed audio ready!"}))
    except Exception as exc:
        dub["error"] = str(exc)
        _push(q, "error", json.dumps({"msg": str(exc)}))
    finally:
        q.put(None)


def _prepare_dub_inputs(job_id: str, lang: str) -> tuple:
    """Validate job state and build (audio_path, raw_segs, seg_to_speaker)."""
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
    speaker_segs = job.get("speaker_segs", [])
    seg_to_speaker: dict = {}
    for entry in speaker_segs:
        if len(entry) == 4:
            s, _e, _t, spk = entry
            for i, (rs, _re, _rt) in enumerate(raw_segs):
                if abs(rs - s) < 0.1:
                    seg_to_speaker[i] = spk
                    break
    return audio_path, raw_segs, seg_to_speaker


def _duck_speech(audio, raw_segs: list):
    """Return a copy of audio with speech windows reduced by −20 dB."""
    ducked = audio
    for start, end, _ in raw_segs:
        s_ms = int(start * 1000)
        e_ms = min(int(end * 1000), len(ducked))
        if s_ms >= e_ms:
            continue
        chunk = ducked[s_ms:e_ms] - 20
        ducked = ducked[:s_ms] + chunk + ducked[e_ms:]
    return ducked


def _overlay_tts(base_audio, raw_segs: list, seg_to_speaker: dict,
                 voice_profiles: dict, default_voice: str, q):
    """Generate per-segment TTS and overlay onto base_audio. Returns new AudioSegment."""
    from pydub import AudioSegment  # already imported in caller, safe to re-import
    AudioSegment.converter = FFMPEG
    result_audio = base_audio
    total = len(raw_segs)
    _push(q, "progress", json.dumps({"msg": f"Generating TTS (0/{total})\u2026", "pct": 15}))

    for i, (start, _end, text) in enumerate(raw_segs):
        pct = 15 + int(75 * (i + 1) / max(total, 1))
        _push(q, "progress", json.dumps({
            "msg": f"Generating speech {i + 1}/{total}\u2026", "pct": pct
        }))
        if not text.strip():
            continue

        spk_id = seg_to_speaker.get(i, "SPEAKER_00")
        profile = voice_profiles.get(spk_id, {})
        voice = profile.get("voice") or default_voice
        rate = profile.get("rate", "+0%")
        pitch = profile.get("pitch", "+0Hz")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tts_path = tf.name
        try:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_tts_generate(text, voice, tts_path, rate, pitch))
            finally:
                loop.close()
            if os.path.exists(tts_path) and os.path.getsize(tts_path) > 0:
                tts_seg = AudioSegment.from_mp3(tts_path)
                result_audio = result_audio.overlay(tts_seg, position=int(start * 1000))
        except Exception as tts_exc:
            # Log and skip the segment rather than aborting the whole dub
            app.logger.warning("TTS failed for segment %d (voice=%s): %s", i, voice, tts_exc)
        finally:
            if os.path.exists(tts_path):
                os.unlink(tts_path)

    return result_audio


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
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

    # Save video persistently (needed for muxing after dubbing)
    job_id = str(uuid.uuid4())
    video_dir = os.path.join(_LOCAL_CACHE, "video")
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{job_id}{ext}")
    f.save(video_path)

    _jobs[job_id] = {"queue": queue.Queue(), "result": None, "error": None,
                     "original_filename": f.filename}
    threading.Thread(
        target=_run_job,
        args=(job_id, video_path, model_size, language, translate_to),
        daemon=True,
    ).start()
    return jsonify(job_id=job_id)


@app.route("/stream/<job_id>", methods=["GET"])
def stream(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error=_ERR_UNKNOWN_JOB), 404

    def generate():
        q = job["queue"]
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return Response(generate(), mimetype=_MIME_SSE, headers=_SSE_HEADERS)


@app.route("/result/<job_id>", methods=["GET"])
def result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error=_ERR_UNKNOWN_JOB), 404
    if job["error"]:
        return jsonify(error=job["error"]), 500
    if not job["result"]:
        return jsonify(error="Not ready yet"), 202
    return jsonify(job["result"])


@app.route("/download/<job_id>/<lang>/<fmt>", methods=["GET"])
def download(job_id: str, lang: str, fmt: str):
    job = _jobs.get(job_id)
    if not job or not job["result"]:
        return jsonify(error=_ERR_NOT_FOUND), 404
    if fmt not in ("srt", "vtt", "txt"):
        return jsonify(error="Invalid format"), 400
    langs = job["result"].get("langs", {})
    if lang not in langs:
        return jsonify(error="Language not available"), 404
    key = "plain" if fmt == "txt" else fmt
    content = langs[lang][key]
    buf = io.BytesIO(content.encode("utf-8"))
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"subtitles_{lang}.{fmt}",
        mimetype="text/plain; charset=utf-8",
    )


@app.route("/dub/<job_id>/<lang>", methods=["POST"])
def start_dub(job_id: str, lang: str):
    job = _jobs.get(job_id)
    if not job or not job.get("result"):
        return jsonify(error="Transcription job not found"), 404
    if lang not in job["result"].get("langs", {}):
        return jsonify(error="Language not available for this job"), 400
    # Accept optional per-speaker voice profiles from JSON body
    voice_profiles = {}
    if request.is_json and request.json:
        voice_profiles = request.json.get("voice_profiles", {})
    dub_id = str(uuid.uuid4())
    _dub_jobs[dub_id] = {"queue": queue.Queue(), "result_path": None, "error": None,
                         "job_id": job_id, "lang": lang}
    threading.Thread(
        target=_dub_job,
        args=(dub_id, job_id, lang, voice_profiles),
        daemon=True,
    ).start()
    return jsonify(dub_id=dub_id)


@app.route("/speakers/<job_id>", methods=["GET"])
def get_speakers(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error=_ERR_UNKNOWN_JOB), 404
    if not job.get("result"):
        return jsonify(error="Not ready yet"), 202
    speakers = job.get("speakers", {})
    return jsonify({"speakers": speakers, "diarization_available": bool(speakers)})


@app.route("/media_info/<job_id>", methods=["GET"])
def media_info(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error=_ERR_UNKNOWN_JOB), 404
    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify(error="Video file not available"), 404
    try:
        info = _run_ffprobe(video_path)
        return jsonify(info)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500


@app.route("/media_info_file", methods=["POST"])
def media_info_file():
    """Run ffprobe on an uploaded file via stdin — no temp file written."""
    f = request.files.get("video")
    if not f:
        return jsonify(error="No file provided"), 400
    data = f.read()
    r = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_streams", "-show_format", "-i", "pipe:0"],
        input=data,
        capture_output=True,
    )
    if r.returncode != 0:
        return jsonify(error="ffprobe error: " + r.stderr.decode(errors="replace")), 500
    return jsonify(json.loads(r.stdout.decode()))


@app.route("/voices/<lang>", methods=["GET"])
def list_voices(lang: str):
    """Return available edge-tts voices for the given language prefix (e.g. 'en', 'es')."""
    async def _fetch():
        import edge_tts
        all_voices = await edge_tts.list_voices()
        return [v for v in all_voices if v["Locale"].lower().startswith(lang.lower())]

    loop = asyncio.new_event_loop()
    try:
        voices = loop.run_until_complete(_fetch())
    finally:
        loop.close()

    return jsonify([{
        "shortName": v["ShortName"],
        "gender": v.get("Gender", ""),
        "friendlyName": v.get("FriendlyName", v["ShortName"]),
        "locale": v["Locale"],
    } for v in voices])


@app.route("/mux/<job_id>/<dub_id>", methods=["POST"])
def start_mux(job_id: str, dub_id: str):
    job = _jobs.get(job_id)
    if not job:
        return jsonify(error="Unknown transcription job"), 404
    dub = _dub_jobs.get(dub_id)
    if not dub or not dub.get("result_path"):
        return jsonify(error="Dub job not complete"), 400
    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify(error="Original video not available for muxing"), 404
    include_original = True
    include_dubbed = True
    output_format = "mp4"
    if request.is_json and request.json:
        include_original = bool(request.json.get("include_original", True))
        include_dubbed = bool(request.json.get("include_dubbed", True))
        output_format = request.json.get("output_format", "mp4")
    if output_format not in ("mp4", "mkv"):
        output_format = "mp4"
    if not include_original and not include_dubbed:
        return jsonify(error="At least one audio track must be selected"), 400

    out_ext = "." + output_format
    audio_bitrate = job.get("source_audio_bitrate", "192k")
    mux_id = str(uuid.uuid4())
    _mux_jobs[mux_id] = {"queue": queue.Queue(), "result_path": None, "error": None}
    threading.Thread(
        target=_mux_job,
        args=(mux_id, video_path, dub["result_path"],
              include_original, include_dubbed, out_ext, audio_bitrate, job_id),
        daemon=True,
    ).start()
    return jsonify(mux_id=mux_id)


@app.route("/stream_mux/<mux_id>", methods=["GET"])
def stream_mux(mux_id: str):
    mux = _mux_jobs.get(mux_id)
    if not mux:
        return jsonify(error=_ERR_UNKNOWN_JOB), 404

    def generate():
        q = mux["queue"]
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return Response(generate(), mimetype=_MIME_SSE, headers=_SSE_HEADERS)


@app.route("/download_muxed/<mux_id>", methods=["GET"])
def download_muxed(mux_id: str):
    mux = _mux_jobs.get(mux_id)
    if not mux:
        return jsonify(error=_ERR_NOT_FOUND), 404
    if mux.get("error"):
        return jsonify(error=mux["error"]), 500
    if not mux.get("result_path") or not os.path.exists(mux["result_path"]):
        return jsonify(error="Not ready"), 202
    ext = Path(mux["result_path"]).suffix
    mime = "video/x-matroska" if ext.lower() == ".mkv" else "video/mp4"
    return send_file(
        mux["result_path"],
        as_attachment=True,
        download_name=f"dubbed_video{ext}",
        mimetype=mime,
    )


@app.route("/stream_dub/<dub_id>", methods=["GET"])
def stream_dub(dub_id: str):
    dub = _dub_jobs.get(dub_id)
    if not dub:
        return jsonify(error=_ERR_UNKNOWN_JOB), 404

    def generate():
        q = dub["queue"]
        while True:
            msg = q.get()
            if msg is None:
                break
            yield msg

    return Response(generate(), mimetype=_MIME_SSE, headers=_SSE_HEADERS)


@app.route("/download_dub/<dub_id>", methods=["GET"])
def download_dub(dub_id: str):
    dub = _dub_jobs.get(dub_id)
    if not dub:
        return jsonify(error=_ERR_NOT_FOUND), 404
    if dub["error"]:
        return jsonify(error=dub["error"]), 500
    path = dub.get("result_path")
    if not path or not os.path.exists(path):
        return jsonify(error="Audio not ready"), 202
    return send_file(path, as_attachment=True, download_name="dubbed_audio.mp3")


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({"output_folder": _config.get("output_folder", "")})


@app.route("/system-info", methods=["GET"])
def system_info():
    return jsonify({"device": _DEVICE, "compute_type": _COMPUTE_TYPE, "label": _DEVICE_LABEL})


@app.route("/browse-folder", methods=["GET"])
def browse_folder():
    """Open a native OS folder-picker dialog and return the selected path.
    Uses a subprocess so tkinter always runs on a fresh main thread.
    Returns {"path": ""} gracefully when a display / tkinter is unavailable
    (headless Linux servers, CI, etc.).
    """
    script = (
        "import sys; "
        "try:\n"
        "    import tkinter as tk\n"
        "    from tkinter import filedialog\n"
        "    root = tk.Tk(); root.withdraw()\n"
        "    try: root.wm_attributes('-topmost', True)\n"
        "    except Exception: pass\n"
        "    path = filedialog.askdirectory(title='Select output folder')\n"
        "    print(path, end='')\n"
        "except Exception as e:\n"
        "    print('__error__:' + str(e), end='', file=sys.stderr)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        path = result.stdout.strip()
        if not path and result.stderr.startswith("__error__:"):
            msg = result.stderr[len("__error__:"):]
            return jsonify({"path": "", "error": msg})
        return jsonify({"path": path})
    except subprocess.TimeoutExpired:
        return jsonify({"path": "", "error": "Dialog timed out"})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"path": "", "error": str(exc)})

@app.route("/config", methods=["POST"])
def set_config():
    if not request.is_json:
        return jsonify(error="JSON required"), 400
    folder = (request.json.get("output_folder") or "").strip()
    if folder and not os.path.isabs(folder):
        return jsonify(error="Please provide an absolute path"), 400
    if folder:
        try:
            os.makedirs(folder, exist_ok=True)
            _test = os.path.join(folder, ".write_test")
            with open(_test, "w") as _f:
                _f.write("ok")
            os.unlink(_test)
        except OSError as exc:
            return jsonify(error=f"Cannot write to folder: {exc}"), 400
    _config["output_folder"] = folder
    msg = f"Saving to: {folder}" if folder else "Auto-save disabled"
    return jsonify(message=msg)


# ── HTML template ──────────────────────────────────────────────────────────────

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Video Subtitler</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;overflow-x:hidden}
header{background:#1e293b;padding:1.25rem 2rem;border-bottom:1px solid #334155;display:flex;align-items:center;gap:.75rem}
header h1{font-size:1.35rem;font-weight:700}
.badge{background:#6366f1;color:#fff;font-size:.7rem;padding:2px 8px;border-radius:999px;text-transform:uppercase;letter-spacing:.05em}
main{flex:1;min-width:0}
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
/* Modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;display:none;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:1.5rem;max-width:740px;width:95%;max-height:85vh;overflow-y:auto}
.modal h3{margin:0 0 1rem;font-size:1.1rem;color:#e2e8f0}
.modal-close{float:right;background:none;border:none;color:#94a3b8;font-size:1.3rem;cursor:pointer;line-height:1}
.stream-cards{display:flex;flex-wrap:wrap;gap:.75rem;margin-top:.5rem}
.stream-card{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:.75rem 1rem;flex:1;min-width:200px}
.stream-card h4{margin:0 0 .5rem;font-size:.78rem;color:#818cf8;text-transform:uppercase;letter-spacing:.05em}
.stream-card p{margin:.15rem 0;font-size:.8rem;color:#94a3b8}
/* Speaker cards */
.spk-grid{display:flex;flex-wrap:wrap;gap:.75rem;margin:.5rem 0}
.spk-card{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:.85rem;flex:1;min-width:230px}
.spk-card h4{margin:0 0 .5rem;font-size:.85rem;color:#c084fc}
.spk-card label{display:block;font-size:.75rem;color:#64748b;margin:.45rem 0 .15rem}
.spk-card select{width:100%;background:#1e293b;color:#e2e8f0;border:1px solid #475569;border-radius:5px;padding:.3rem .5rem;font-size:.78rem}
.spk-card .slider-row{display:flex;align-items:center;gap:.4rem}
.spk-card input[type=range]{flex:1;accent-color:#818cf8}
.spk-card .slider-val{min-width:38px;font-size:.75rem;color:#94a3b8;text-align:right}
.gender-toggle{display:flex;gap:.3rem;margin-bottom:.35rem}
.gender-btn{background:#1e293b;border:1px solid #475569;color:#94a3b8;border-radius:5px;padding:.2rem .6rem;font-size:.75rem;cursor:pointer}
.gender-btn.active{background:#312e81;border-color:#6366f1;color:#e0e7ff}
/* Mux */
.mux-box{background:#0f172a;border:1px solid #334155;border-radius:8px;padding:.85rem;margin-top:.75rem}
.mux-box h4{margin:0 0 .5rem;font-size:.82rem;color:#34d399}
.mux-radio{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:.5rem}
.mux-radio label{display:flex;align-items:center;gap:.35rem;font-size:.8rem;color:#94a3b8;cursor:pointer}
/* Layout with sidebar */
.app-layout{display:flex;align-items:flex-start;gap:0;max-width:1280px;margin:2rem auto;padding:0 1rem}
.app-main{flex:1;min-width:0}
/* Log sidebar */
.log-sidebar{width:280px;flex-shrink:0;position:sticky;top:1rem;margin-left:1rem;transition:width .25s}
.log-sidebar.collapsed{width:32px}
.log-toggle{background:#1e293b;border:1px solid #334155;border-radius:8px 8px 0 0;padding:.4rem .6rem;display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none}
.log-toggle span{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;white-space:nowrap;overflow:hidden}
.log-toggle svg{flex-shrink:0;transition:transform .25s}
.log-sidebar.collapsed .log-toggle svg{transform:rotate(180deg)}
.log-sidebar.collapsed .log-toggle span{display:none}
.log-body{background:#0a0f1a;border:1px solid #334155;border-top:none;border-radius:0 0 8px 8px;height:calc(100vh - 120px);overflow-y:auto;display:flex;flex-direction:column;padding:.5rem}
.log-sidebar.collapsed .log-body{display:none}
.log-entry{font-size:.72rem;font-family:monospace;line-height:1.5;padding:.1rem 0;border-bottom:1px solid #1e293b;word-break:break-word}
.log-entry.info{color:#94a3b8}
.log-entry.ok{color:#4ade80}
.log-entry.err{color:#f87171}
.log-entry.prog{color:#818cf8}
.log-clear{background:none;border:none;font-size:.65rem;color:#475569;cursor:pointer;align-self:flex-end;margin-bottom:.25rem;flex-shrink:0}
.log-clear:hover{color:#94a3b8}
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
  <span id="device-badge" class="badge" style="background:#1e3a5f;color:#7dd3fc;display:none"></span>
</header>

<div class="app-layout">
<main class="app-main">
  <!-- Output Settings -->
  <div class="card">
    <h2>&#9881; Output Settings <span style="font-weight:400;color:#475569;font-size:.82rem">(optional)</span></h2>
    <div class="row" style="align-items:flex-end;gap:.75rem">
      <div class="field" style="flex:2;min-width:200px">
        <label for="output-folder">Auto-save subtitles &amp; audio to folder</label>
        <input type="text" id="output-folder" placeholder="/absolute/path/to/output  (leave blank to download manually)"
               style="background:#0f172a;border:1px solid #475569;color:#e2e8f0;border-radius:6px;padding:.5rem .75rem;font-size:.88rem;width:100%"/>
      </div>
      <button class="btn" id="browse-folder-btn" title="Pick a folder" style="padding:.5rem 1rem;font-size:.85rem;flex-shrink:0;background:#1e293b;border:1px solid #475569">&#128193; Browse…</button>
      <button class="btn" id="set-folder-btn" style="padding:.5rem 1.1rem;font-size:.85rem;flex-shrink:0">Set</button>
    </div>
    <div id="folder-status" style="font-size:.76rem;margin-top:.35rem;color:#64748b"></div>
  </div>

  <!-- Step 1 -->
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.75rem">
      <h2 style="margin:0">1 &mdash; Upload video</h2>
      <button class="btn" id="media-info-btn" disabled style="background:#0c4a6e;border:1px solid #0369a1;color:#7dd3fc;padding:.35rem .9rem;font-size:.8rem">&#128249; Media Info</button>
    </div>
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
    <h2 style="margin-bottom:.5rem">6 &mdash; Dub Audio <span style="color:#94a3b8;font-weight:400">(AI voices per speaker)</span></h2>
    <p class="dub-note" style="margin-top:.7rem">Background music is preserved (briefly ducked during dialogue). Each detected speaker gets its own neural voice. Fine-tune rate and pitch &plusmn;5% to match the original character. Internet required for voice synthesis.</p>
    <div class="row" style="align-items:flex-end;margin-top:.25rem">
      <div class="field">
        <label for="dub-lang-sel">Target language</label>
        <select id="dub-lang-sel"></select>
      </div>
      <button class="btn" id="load-speakers-btn" style="background:#0c4a6e;border:1px solid #0369a1;color:#7dd3fc">Load Speaker Profiles</button>
    </div>
    <div id="spk-section" style="display:none;margin-top:.75rem">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.25rem">
        <span style="font-size:.83rem;color:#94a3b8">Detected speakers</span>
        <span id="spk-loading" style="font-size:.78rem;color:#64748b"></span>
      </div>
      <div class="spk-grid" id="spk-grid"></div>
    </div>
    <div style="margin-top:.75rem">
      <button class="btn" id="dub-btn">Generate dubbed audio</button>
    </div>
    <div id="dub-prog-wrap" style="display:none;margin-top:1rem">
      <div class="progress-bar-bg"><div class="progress-bar" id="dub-prog-bar"></div></div>
      <div class="progress-msg" id="dub-prog-msg">Starting&hellip;</div>
    </div>
    <div class="error-box" id="dub-error-box"></div>
    <div id="dub-result-area" style="display:none;margin-top:1rem">
      <p class="success-msg">&checkmark; Dubbed audio ready</p>
      <div class="dl-bar">
        <a id="dub-dl-btn" class="dl-btn" href="#">&#11015; Download dubbed audio (.mp3)</a>
        <button class="btn" id="mux-btn" style="background:#064e3b;border-color:#065f46;padding:.4rem 1rem;font-size:.82rem">&#127916; Mux into video</button>
      </div>
      <div class="mux-box" id="mux-box" style="display:none">
        <h4>&#127916; Bundle Options</h4>
        <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:.65rem">
          <div>
            <span style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;display:block;margin-bottom:.3rem">Container</span>
            <div class="mux-radio">
              <label><input type="radio" name="mux-fmt" value="mp4" checked> MP4</label>
              <label><input type="radio" name="mux-fmt" value="mkv"> MKV</label>
            </div>
          </div>
          <div>
            <span style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;display:block;margin-bottom:.3rem">Audio tracks</span>
            <div style="display:flex;flex-direction:column;gap:.3rem">
              <label style="display:flex;align-items:center;gap:.4rem;font-size:.8rem;color:#94a3b8;cursor:pointer"><input type="checkbox" id="mux-orig" checked> Keep original audio track(s)</label>
              <label style="display:flex;align-items:center;gap:.4rem;font-size:.8rem;color:#94a3b8;cursor:pointer"><input type="checkbox" id="mux-dubbed" checked> Include dubbed audio track</label>
            </div>
          </div>
        </div>
        <button class="btn" id="mux-go-btn">Build video</button>
        <div id="mux-prog-wrap" style="display:none;margin-top:.75rem">
          <div class="progress-bar-bg"><div class="progress-bar" id="mux-prog-bar"></div></div>
          <div class="progress-msg" id="mux-prog-msg">Muxing&hellip;</div>
        </div>
        <div class="error-box" id="mux-error-box"></div>
        <div id="mux-result-area" style="display:none;margin-top:.75rem">
          <p class="success-msg">&checkmark; Video ready</p>
          <a id="mux-dl-btn" class="dl-btn" href="#">&#11015; Download muxed video</a>
        </div>
      </div>
    </div>
  </div>
</main>

<!-- Log sidebar -->
<aside class="log-sidebar" id="log-sidebar">
  <div class="log-toggle" id="log-toggle">
    <span>&#128221; Activity Log</span>
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
  </div>
  <div class="log-body" id="log-body">
    <button class="log-clear" id="log-clear">clear</button>
  </div>
</aside>
</div>

<!-- Media Info Modal -->
<div class="modal-overlay" id="media-modal">
  <div class="modal">
    <button class="modal-close" onclick="closeMediaModal()">&times;</button>
    <h3>&#128249; Media Info</h3>
    <div id="media-format-info" style="font-size:.82rem;color:#64748b;margin-bottom:.5rem"></div>
    <div class="stream-cards" id="media-streams"></div>
    <div style="margin-top:1rem;font-size:.75rem;color:#475569">Stream details from ffprobe</div>
  </div>
</div>
<script>
// ── Activity Log ─────────────────────────────────────────────────────────────
const logSidebar = document.getElementById("log-sidebar");
const logBody    = document.getElementById("log-body");
const logToggle  = document.getElementById("log-toggle");
const logClear   = document.getElementById("log-clear");

logToggle.addEventListener("click", () => logSidebar.classList.toggle("collapsed"));
logClear.addEventListener("click", () => {
  [...logBody.querySelectorAll(".log-entry")].forEach(el => el.remove());
});

function log(msg, type = "info") {
  const now = new Date();
  const ts  = now.toTimeString().slice(0,8);
  const el  = document.createElement("div");
  el.className = "log-entry " + type;
  el.textContent = "[" + ts + "] " + msg;
  logBody.appendChild(el);
  logBody.scrollTop = logBody.scrollHeight;
}

// ── Output folder config ──────────────────────────────────────────────
const outputFolderInput = document.getElementById("output-folder");
const setFolderBtn      = document.getElementById("set-folder-btn");
const browseFolderBtn   = document.getElementById("browse-folder-btn");
const folderStatus      = document.getElementById("folder-status");

browseFolderBtn.addEventListener("click", async () => {
  browseFolderBtn.disabled = true;
  browseFolderBtn.textContent = "\u23F3 Opening\u2026";
  folderStatus.textContent = "Waiting for folder picker\u2026";
  folderStatus.style.color = "#94a3b8";
  try {
    const r = await fetch("/browse-folder");
    const j = await r.json();
    if (j.path) {
      outputFolderInput.value = j.path;
      folderStatus.textContent = "";
    } else {
      folderStatus.textContent = j.error ? "Error: " + j.error : "No folder selected.";
      folderStatus.style.color = "#f87171";
    }
  } catch (e) {
    folderStatus.textContent = "Error: " + e.message;
    folderStatus.style.color = "#f87171";
  } finally {
    browseFolderBtn.disabled = false;
    browseFolderBtn.textContent = "\uD83D\uDCC1 Browse\u2026";
  }
});

(async () => {
  try {
    const r = await fetch("/config");
    if (r.ok) {
      const j = await r.json();
      if (j.output_folder) {
        outputFolderInput.value = j.output_folder;
        folderStatus.textContent = "Active: " + j.output_folder;
        folderStatus.style.color = "#4ade80";
      }
    }
  } catch (_) {}
  try {
    const sr = await fetch("/system-info");
    if (sr.ok) {
      const si = await sr.json();
      const badge = document.getElementById("device-badge");
      badge.textContent = si.label;
      badge.style.background = si.device === "cuda" ? "#14532d" : "#1e3a5f";
      badge.style.color      = si.device === "cuda" ? "#86efac" : "#7dd3fc";
      badge.style.display    = "";
    }
  } catch (_) {}
})();

setFolderBtn.addEventListener("click", async () => {
  const folder = outputFolderInput.value.trim();
  let resp;
  try {
    resp = await fetch("/config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({output_folder: folder}),
    });
  } catch (e) {
    folderStatus.textContent = "Error: " + e.message;
    folderStatus.style.color = "#f87171";
    return;
  }
  const j = await resp.json();
  if (resp.ok) {
    folderStatus.textContent = j.message;
    folderStatus.style.color = "#4ade80";
    log("Output folder: " + (folder || "disabled"), "ok");
  } else {
    folderStatus.textContent = "Error: " + j.error;
    folderStatus.style.color = "#f87171";
    log("Folder error: " + j.error, "err");
  }
});

// ── App state ─────────────────────────────────────────────────────────────────
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
  mediaInfoBtn.disabled = !f;
  if (f) log("File selected: " + f.name + " (" + (f.size/1024/1024).toFixed(1) + " MB)");
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
  log("Upload complete — job " + jobId.slice(0,8));
  log("Transcription started (model: " + document.getElementById("model-sel").value + ")");
  listenProgress(jobId);
});

function listenProgress(jid) {
  const es = new EventSource("/stream/" + jid);
  es.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    progMsg.textContent = d.msg;
    progBar.style.width = (d.pct || 0) + "%";
    log(d.msg, "prog");
  });
  es.addEventListener("done", async () => {
    es.close();
    progBar.style.width = "100%";
    progMsg.textContent = "Done!";
    log("Transcription done", "ok");
    const r = await fetch("/result/" + jid);
    const data = await r.json();
    showResults(jid, data);
  });
  es.addEventListener("error", e => {
    es.close();
    let msg;
    try { msg = JSON.parse(e.data).msg; } catch(_) { msg = "Transcription failed."; }
    showError(msg);
    log("Transcription error: " + msg, "err");
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
  log("Results: " + data.segment_count + " segs, lang=" + (data.detected_language_name || data.detected_language) + ", " + Object.keys(data.langs).length + " version(s)", "ok");
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
const dubCard        = document.getElementById("dub-card");
const dubLangSel     = document.getElementById("dub-lang-sel");
const dubBtn         = document.getElementById("dub-btn");
const dubProgWrap    = document.getElementById("dub-prog-wrap");
const dubProgBar     = document.getElementById("dub-prog-bar");
const dubProgMsg     = document.getElementById("dub-prog-msg");
const dubErrorBox    = document.getElementById("dub-error-box");
const dubResultArea  = document.getElementById("dub-result-area");
const dubDlBtn       = document.getElementById("dub-dl-btn");
const loadSpeakersBtn = document.getElementById("load-speakers-btn");
const spkSection     = document.getElementById("spk-section");
const spkGrid        = document.getElementById("spk-grid");
const spkLoading     = document.getElementById("spk-loading");
const muxBtn         = document.getElementById("mux-btn");
const muxBox         = document.getElementById("mux-box");
const muxGoBtn       = document.getElementById("mux-go-btn");
const muxProgWrap    = document.getElementById("mux-prog-wrap");
const muxProgBar     = document.getElementById("mux-prog-bar");
const muxProgMsg     = document.getElementById("mux-prog-msg");
const muxErrorBox    = document.getElementById("mux-error-box");
const muxResultArea  = document.getElementById("mux-result-area");
const muxDlBtn       = document.getElementById("mux-dl-btn");
const mediaInfoBtn   = document.getElementById("media-info-btn");
const mediaModal     = document.getElementById("media-modal");
const mediaStreams    = document.getElementById("media-streams");
const mediaFormatInfo = document.getElementById("media-format-info");

const dubLangNames = {
  en:"English",es:"Spanish",fr:"French",de:"German",it:"Italian",pt:"Portuguese",
  nl:"Dutch",pl:"Polish",ru:"Russian",ja:"Japanese",ko:"Korean",zh:"Chinese",
  ar:"Arabic",tr:"Turkish",sv:"Swedish",da:"Danish",fi:"Finnish",no:"Norwegian",he:"Hebrew"
};

let currentDubId = null;

function initDubCard(data) {
  dubLangSel.innerHTML = "";
  for (const code of Object.keys(data.langs)) {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = (dubLangNames[code] || code) +
                      (code === data.detected_language ? " (original)" : "");
    dubLangSel.appendChild(opt);
  }
  dubProgWrap.style.display   = "none";
  dubErrorBox.style.display   = "none";
  dubResultArea.style.display = "none";
  spkSection.style.display    = "none";
  muxBox.style.display        = "none";
  dubBtn.disabled = false;
  dubCard.style.display = "block";
}

// ── Media Info Modal ─────────────────────────────────────────────────────────
mediaInfoBtn.addEventListener("click", openMediaModal);
mediaModal.addEventListener("click", e => { if (e.target === mediaModal) closeMediaModal(); });

async function openMediaModal() {
  if (!selectedFile && !jobId) return;
  mediaFormatInfo.textContent = "Loading\u2026";
  mediaStreams.innerHTML = "";
  mediaModal.classList.add("open");
  try {
    let resp;
    if (jobId) {
      resp = await fetch("/media_info/" + jobId);
    } else {
      const fd = new FormData();
      fd.append("video", selectedFile);
      resp = await fetch("/media_info_file", { method: "POST", body: fd });
    }
    if (!resp.ok) throw new Error("Media info unavailable");
    const info = await resp.json();
    renderMediaInfo(info);
  } catch (e) { mediaFormatInfo.textContent = "Error: " + e.message; }
}

function closeMediaModal() { mediaModal.classList.remove("open"); }

function renderMediaInfo(info) {
  const fmt = info.format || {};
  const dur = parseFloat(fmt.duration || 0);
  const sz  = parseInt(fmt.size || 0);
  mediaFormatInfo.innerHTML =
    "<strong>" + (fmt.format_long_name || fmt.format_name || "Unknown") + "</strong>" +
    (dur > 0 ? " &middot; " + (dur / 60).toFixed(1) + " min" : "") +
    (sz  > 0 ? " &middot; " + (sz / 1024 / 1024).toFixed(1) + " MB" : "") +
    (fmt.bit_rate ? " &middot; " + Math.round(parseInt(fmt.bit_rate) / 1000) + " kbps" : "");
  mediaStreams.innerHTML = "";
  for (const s of (info.streams || [])) {
    const card = document.createElement("div");
    card.className = "stream-card";
    if (s.codec_type === "video") {
      card.innerHTML = "<h4>&#127916; Video #" + s.index + "</h4>" +
        "<p>Codec: " + (s.codec_name || "?") + "</p>" +
        "<p>Resolution: " + (s.width || "?") + "\xd7" + (s.height || "?") + "</p>" +
        "<p>Frame rate: " + (s.avg_frame_rate || s.r_frame_rate || "?") + "</p>" +
        "<p>Pixel fmt: " + (s.pix_fmt || "?") + "</p>";
    } else if (s.codec_type === "audio") {
      card.innerHTML = "<h4>&#127925; Audio #" + s.index + "</h4>" +
        "<p>Codec: " + (s.codec_name || "?") + "</p>" +
        "<p>Channels: " + (s.channels || "?") + " (" + (s.channel_layout || "?") + ")</p>" +
        "<p>Sample rate: " + (s.sample_rate || "?") + " Hz</p>" +
        "<p>Bitrate: " + (s.bit_rate ? Math.round(parseInt(s.bit_rate)/1000) + " kbps" : "?") + "</p>";
    } else {
      card.innerHTML = "<h4>Track #" + s.index + " (" + s.codec_type + ")</h4>" +
        "<p>Codec: " + (s.codec_name || "?") + "</p>";
    }
    mediaStreams.appendChild(card);
  }
}

// ── Speaker profiles ──────────────────────────────────────────────────────────
loadSpeakersBtn.addEventListener("click", async () => {
  if (!jobId) return;
  const lang = dubLangSel.value;
  spkSection.style.display = "block";
  spkLoading.textContent = "Loading\u2026";
  spkGrid.innerHTML = "";
  log("Loading speaker profiles for lang=" + lang);
  try {
    const [spkResp, voiceResp] = await Promise.all([
      fetch("/speakers/" + jobId),
      fetch("/voices/" + lang),
    ]);
    const spkData = await spkResp.json();
    const voices  = voiceResp.ok ? await voiceResp.json() : [];
    const spks    = spkData.speakers || {};
    const nSpk    = Object.keys(spks).length;
    spkLoading.textContent = nSpk
      ? nSpk + " speaker(s) detected"
      : "Single-speaker mode";
    renderSpeakerCards(spks, voices, resultData ? resultData.detected_language : null);
    log((nSpk || 1) + " speaker(s) loaded, " + voices.length + " voice(s) available", "ok");
  } catch (e) {
    spkLoading.textContent = "Could not load profiles: " + e.message;
    log("Speaker load error: " + e.message, "err");
  }
});

function renderSpeakerCards(speakers, voices, srcLang) {
  spkGrid.innerHTML = "";
  if (srcLang) {
    const _noteEl = document.createElement("p");
    _noteEl.style.cssText = "width:100%;font-size:.75rem;color:#475569;margin:.1rem 0 .5rem";
    const _lnNames = {en:"English",es:"Spanish",fr:"French",de:"German",it:"Italian",pt:"Portuguese",nl:"Dutch",pl:"Polish",ru:"Russian",ja:"Japanese",ko:"Korean",zh:"Chinese",ar:"Arabic",tr:"Turkish",sv:"Swedish",da:"Danish",fi:"Finnish",no:"Norwegian",he:"Hebrew"};
    _noteEl.innerHTML = "Source language: <strong style='color:#94a3b8'>" + (_lnNames[srcLang] || srcLang) + "</strong> \u2014 speed &amp; pitch are pre-set from acoustic analysis. Choose a voice accent to match.";
    spkGrid.appendChild(_noteEl);
  }
  let keys = Object.keys(speakers).sort();
  if (!keys.length) {
    keys = ["SPEAKER_00"];
    speakers = {"SPEAKER_00": {gender_hint:"U", segment_count:0, total_duration:0}};
  }
  const maleVoices   = voices.filter(v => v.gender === "Male");
  const femaleVoices = voices.filter(v => v.gender === "Female");

  for (const spk of keys) {
    const info   = speakers[spk];
    const gender = info.gender_hint || "U";
    const initVoices = gender === "F" ? femaleVoices : maleVoices;
    const voiceList  = (initVoices.length ? initVoices : voices);

    const card = document.createElement("div");
    card.className  = "spk-card";
    card.dataset.spk    = spk;
    card.dataset.gender = gender === "F" ? "F" : "M";

    const meta = info.segment_count
      ? " \u2014 " + info.segment_count + " segs \u00b7 " + info.total_duration + "s" +
        (info.median_f0 ? " \u00b7 F0: " + info.median_f0 + " Hz" : "") +
        (info.speaking_wps ? " \u00b7 " + info.speaking_wps + " wps" : "")
      : "";

    const mkOpts = arr => arr.map(v =>
      "<option value='" + v.shortName + "'>" + v.friendlyName + " (" + v.gender[0] + ")</option>"
    ).join("");

    const sugRate  = typeof info.suggested_rate  === "number" ? info.suggested_rate  : 0;
    const sugPitch = typeof info.suggested_pitch === "number" ? info.suggested_pitch : 0;
    const rateDisp  = (sugRate  >= 0 ? "+" : "") + sugRate  + "%";
    const pitchDisp = (sugPitch >= 0 ? "+" : "") + sugPitch + "Hz";

    card.innerHTML =
      "<h4>" + spk.replace("SPEAKER_","Speaker ") +
        "<span style='font-weight:400;color:#64748b;font-size:.72rem'>" + meta + "</span></h4>" +
      "<div class='gender-toggle'>" +
        "<button class='gender-btn" + (gender !== "F" ? " active" : "") + "' data-g='M'>&#9794; Male</button>" +
        "<button class='gender-btn" + (gender === "F" ? " active" : "") + "' data-g='F'>&#9792; Female</button>" +
      "</div>" +
      "<label>Voice</label>" +
      "<select class='voice-sel'>" + mkOpts(voiceList) + "</select>" +
      "<label>Speed <span class='slider-val' id='rate-val-" + spk + "'>" + rateDisp + "</span></label>" +
      "<div class='slider-row'><input type='range' min='-20' max='20' step='2' value='" + sugRate + "' class='rate-sl' id='rate-" + spk + "'></div>" +
      "<label>Pitch <span class='slider-val' id='pitch-val-" + spk + "'>" + pitchDisp + "</span></label>" +
      "<div class='slider-row'><input type='range' min='-10' max='10' step='2' value='" + sugPitch + "' class='pitch-sl' id='pitch-" + spk + "'></div>";

    spkGrid.appendChild(card);

    const voiceSel = card.querySelector(".voice-sel");
    card.querySelectorAll(".gender-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        card.querySelectorAll(".gender-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const filtered = btn.dataset.g === "F" ? femaleVoices : maleVoices;
        voiceSel.innerHTML = mkOpts(filtered.length ? filtered : voices);
      });
    });
    card.querySelector(".rate-sl").addEventListener("input", ev => {
      const v = parseInt(ev.target.value);
      card.querySelector("#rate-val-" + spk).textContent = (v >= 0 ? "+" : "") + v + "%";
    });
    card.querySelector(".pitch-sl").addEventListener("input", ev => {
      const v = parseInt(ev.target.value);
      card.querySelector("#pitch-val-" + spk).textContent = (v >= 0 ? "+" : "") + v + "Hz";
    });
  }
}

function collectVoiceProfiles() {
  const profiles = {};
  for (const card of spkGrid.querySelectorAll(".spk-card")) {
    const spk   = card.dataset.spk;
    const voice = card.querySelector(".voice-sel").value;
    const rate  = parseInt(card.querySelector(".rate-sl").value);
    const pitch = parseInt(card.querySelector(".pitch-sl").value);
    profiles[spk] = {
      voice,
      rate:  (rate  >= 0 ? "+" : "") + rate  + "%",
      pitch: (pitch >= 0 ? "+" : "") + pitch + "Hz",
    };
  }
  return profiles;
}

// ── Generate dub ──────────────────────────────────────────────────────────────
dubBtn.addEventListener("click", async () => {
  const lang = dubLangSel.value;
  if (!jobId || !lang) return;
  dubErrorBox.style.display   = "none";
  dubResultArea.style.display = "none";
  muxBox.style.display        = "none";
  dubProgWrap.style.display   = "block";
  dubProgBar.style.width = "0%";
  dubProgMsg.textContent = "Starting\u2026";
  dubBtn.disabled = true;

  const voiceProfiles = collectVoiceProfiles();
  log("Starting dub — lang=" + lang + ", " + Object.keys(voiceProfiles).length + " speaker profile(s)");

  let resp;
  try {
    resp = await fetch("/dub/" + jobId + "/" + lang, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({voice_profiles: voiceProfiles}),
    });
  } catch (e) { showDubError("Request failed: " + e.message); return; }
  if (!resp.ok) {
    const j = await resp.json().catch(() => ({}));
    showDubError(j.error || "Failed to start dub job");
    return;
  }
  const { dub_id } = await resp.json();
  currentDubId = dub_id;

  const es = new EventSource("/stream_dub/" + dub_id);
  es.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    dubProgMsg.textContent  = d.msg;
    dubProgBar.style.width  = (d.pct || 0) + "%";
    log(d.msg, "prog");
  });
  es.addEventListener("done", () => {
    es.close();
    dubProgBar.style.width = "100%";
    dubProgMsg.textContent = "Done!";
    const name = dubLangNames[lang] || lang;
    dubDlBtn.textContent = "\u2b07 Download dubbed audio \u2014 " + name + " (.mp3)";
    dubDlBtn.href        = "/download_dub/" + dub_id;
    dubResultArea.style.display = "block";
    dubBtn.disabled = false;
    log("Dub complete — " + (dubLangNames[lang] || lang), "ok");
  });
  es.addEventListener("error", e => {
    es.close();
    let msg; try { msg = JSON.parse(e.data).msg; } catch(_) { msg = "Dubbing failed."; }
    showDubError(msg);
    log("Dub error: " + msg, "err");
  });
});

function showDubError(msg) {
  dubProgWrap.style.display = "none";
  dubErrorBox.textContent   = msg;
  dubErrorBox.style.display = "block";
  dubBtn.disabled = false;
}

// ── Mux track ─────────────────────────────────────────────────────────────────
muxBtn.addEventListener("click", () => {
  muxBox.style.display = muxBox.style.display === "none" ? "block" : "none";
});

muxGoBtn.addEventListener("click", async () => {
  if (!jobId || !currentDubId) return;
  const includeOriginal = document.getElementById("mux-orig").checked;
  const includeDubbed   = document.getElementById("mux-dubbed").checked;
  const outputFmt       = document.querySelector("input[name='mux-fmt']:checked").value;
  if (!includeOriginal && !includeDubbed) {
    showMuxError("Select at least one audio track.");
    return;
  }
  muxProgWrap.style.display   = "block";
  muxProgBar.style.width      = "0%";
  muxProgMsg.textContent      = "Starting\u2026";
  muxErrorBox.style.display   = "none";
  muxResultArea.style.display = "none";
  muxGoBtn.disabled = true;
  log("Bundle: " + outputFmt.toUpperCase() + (includeOriginal ? " +orig" : "") + (includeDubbed ? " +dub" : ""));

  let resp;
  try {
    resp = await fetch("/mux/" + jobId + "/" + currentDubId, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({include_original: includeOriginal, include_dubbed: includeDubbed, output_format: outputFmt}),
    });
  } catch (e) { showMuxError("Request failed: " + e.message); return; }
  if (!resp.ok) {
    const j = await resp.json().catch(() => ({}));
    showMuxError(j.error || "Failed to start bundle");
    return;
  }
  const { mux_id } = await resp.json();

  const es = new EventSource("/stream_mux/" + mux_id);
  es.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    muxProgMsg.textContent = d.msg;
    muxProgBar.style.width = (d.pct || 50) + "%";
    log(d.msg, "prog");
  });
  es.addEventListener("done", () => {
    es.close();
    muxProgBar.style.width      = "100%";
    muxProgMsg.textContent      = "Done!";
    muxDlBtn.href               = "/download_muxed/" + mux_id;
    muxDlBtn.textContent        = "\u2b07 Download " + outputFmt.toUpperCase() + " video";
    muxResultArea.style.display = "block";
    muxGoBtn.disabled = false;
    log("Bundle complete \u2014 " + outputFmt.toUpperCase(), "ok");
  });
  es.addEventListener("error", e => {
    es.close();
    let msg; try { msg = JSON.parse(e.data).msg; } catch(_) { msg = "Bundle failed."; }
    showMuxError(msg);
    log("Bundle error: " + msg, "err");
  });
});

function showMuxError(msg) {
  muxProgWrap.style.display = "none";
  muxErrorBox.textContent   = msg;
  muxErrorBox.style.display = "block";
  muxGoBtn.disabled = false;
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    _port = int(os.environ.get("PORT", 5001))
    _url  = f"http://127.0.0.1:{_port}"
    # Open the browser slightly after the server starts so the socket is ready.
    # On Linux headless servers webbrowser.open may silently do nothing, which is fine.
    threading.Timer(1.2, lambda: webbrowser.open(_url)).start()
    app.run(host="127.0.0.1", port=_port, threaded=True, debug=False)
