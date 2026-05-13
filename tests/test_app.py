"""
Tests for Video Subtitler (app.py).

All heavy dependencies (Whisper, argostranslate, edge-tts, pydub, ffmpeg) are
mocked so the suite runs instantly without any models or binaries installed.
"""

import io
import json
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── stub heavy third-party modules before importing app ──────────────────────
# faster_whisper
_fw = MagicMock()
sys.modules.setdefault("faster_whisper", _fw)

# argostranslate sub-packages
for _mod in ("argostranslate", "argostranslate.package", "argostranslate.translate"):
    sys.modules.setdefault(_mod, MagicMock())

import app  # noqa: E402  (must come after stubbing)
from app import (  # noqa: E402
    _build_srt,
    _build_vtt,
    _fmt_srt,
    _fmt_vtt,
    EDGE_TTS_VOICES,
    SUPPORTED_EXTENSIONS,
    MODEL_SIZES,
)


# Fixtures

@pytest.fixture()
def client():
    app.app.config["TESTING"] = True
    app.app.config["WTF_CSRF_ENABLED"] = False
    with app.app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_jobs():
    """Wipe job stores between tests to prevent state leakage."""
    app._jobs.clear()
    app._dub_jobs.clear()
    yield
    app._jobs.clear()
    app._dub_jobs.clear()


def _make_job(result=None, error=None, audio_path=None):
    """Helper: insert a synthetic finished job into _jobs and return its ID."""
    jid = str(uuid.uuid4())
    app._jobs[jid] = {
        "queue": queue.Queue(),
        "result": result,
        "error": error,
        "audio_path": audio_path,
    }
    return jid


def _make_dub(result_path=None, error=None):
    """Helper: insert a synthetic dub job into _dub_jobs and return its ID."""
    did = str(uuid.uuid4())
    app._dub_jobs[did] = {
        "queue": queue.Queue(),
        "result_path": result_path,
        "error": error,
    }
    return did


_SAMPLE_RESULT = {
    "detected_language": "en",
    "detected_language_name": "English",
    "segment_count": 2,
    "langs": {
        "en": {
            "plain": "Hello world\nHow are you",
            "srt": "1\n00:00:01,000 --> 00:00:02,000\nHello world\n\n",
            "vtt": "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello world\n\n",
        }
    },
    "segments": {
        "en": [(1.0, 2.0, "Hello world"), (3.0, 4.0, "How are you")]
    },
}


# Unit tests — subtitle formatters

class TestFmtSrt:
    def test_zero(self):
        assert _fmt_srt(0.0) == "00:00:00,000"

    def test_simple(self):
        assert _fmt_srt(3661.5) == "01:01:01,500"

    def test_millisecond_rounding(self):
        # 1.999 → 999 ms
        assert _fmt_srt(1.999) == "00:00:01,999"

    def test_exactly_one_hour(self):
        assert _fmt_srt(3600.0) == "01:00:00,000"


class TestFmtVtt:
    def test_separator_is_dot(self):
        ts = _fmt_vtt(61.123)
        assert "." in ts
        assert "," not in ts

    def test_zero(self):
        assert _fmt_vtt(0.0) == "00:00:00.000"

    def test_simple(self):
        assert _fmt_vtt(3661.5) == "01:01:01.500"


class TestBuildSrt:
    def test_empty(self):
        assert _build_srt([]) == ""

    def test_single_segment(self):
        out = _build_srt([(1.0, 2.5, "Hello")])
        assert "1\n" in out
        assert "00:00:01,000 --> 00:00:02,500" in out
        assert "Hello" in out

    def test_numbering(self):
        segs = [(0.0, 1.0, "A"), (1.0, 2.0, "B"), (2.0, 3.0, "C")]
        out = _build_srt(segs)
        assert "\n1\n" in ("\n" + out)
        assert "\n2\n" in out
        assert "\n3\n" in out

    def test_segments_separated_by_blank_line(self):
        out = _build_srt([(0.0, 1.0, "A"), (1.0, 2.0, "B")])
        # Each segment ends with a blank line
        assert "\n\n" in out


class TestBuildVtt:
    def test_starts_with_webvtt(self):
        out = _build_vtt([])
        assert out.startswith("WEBVTT")

    def test_single_segment(self):
        out = _build_vtt([(0.5, 1.5, "Hi")])
        assert "00:00:00.500 --> 00:00:01.500" in out
        assert "Hi" in out

    def test_no_sequence_numbers(self):
        out = _build_vtt([(0.0, 1.0, "X")])
        # SRT has sequence numbers; VTT should not
        assert "\n1\n" not in out


# Unit tests — constants / config

class TestConfig:
    def test_supported_extensions_all_lowercase(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")

    def test_model_sizes_order(self):
        assert MODEL_SIZES[0] == "tiny"
        assert MODEL_SIZES[-1] == "large-v3"

    def test_edge_tts_voices_covered(self):
        expected_langs = {"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
                          "ja", "ko", "zh", "ar", "tr", "sv", "da", "fi", "no", "he"}
        assert expected_langs == set(EDGE_TTS_VOICES.keys())

    def test_edge_tts_voices_are_neural(self):
        for voice in EDGE_TTS_VOICES.values():
            assert "Neural" in voice, f"{voice} is not a Neural voice"


# Route: GET /

class TestIndexRoute:
    def test_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_contains_title(self, client):
        r = client.get("/")
        assert b"Video Subtitler" in r.data

    def test_post_not_allowed(self, client):
        r = client.post("/")
        assert r.status_code == 405


# Route: POST /transcribe — validation

class TestTranscribeValidation:
    def test_no_file(self, client):
        r = client.post("/transcribe", data={})
        assert r.status_code == 400
        assert b"No file" in r.data

    def test_unsupported_extension(self, client):
        data = {"video": (io.BytesIO(b"fake"), "clip.xyz")}
        r = client.post("/transcribe", data=data, content_type="multipart/form-data")
        assert r.status_code == 400
        assert b"Unsupported" in r.data

    def test_invalid_model(self, client):
        data = {
            "video": (io.BytesIO(b"fake"), "clip.mp4"),
            "model": "huge",
            "language": "auto",
        }
        r = client.post("/transcribe", data=data, content_type="multipart/form-data")
        assert r.status_code == 400
        assert b"Invalid model" in r.data

    def test_invalid_language(self, client):
        data = {
            "video": (io.BytesIO(b"fake"), "clip.mp4"),
            "model": "base",
            "language": "klingon",
        }
        r = client.post("/transcribe", data=data, content_type="multipart/form-data")
        assert r.status_code == 400
        assert b"Invalid language" in r.data

    def test_unknown_translate_languages_silently_dropped(self, client):
        """translate_to values not in TRANSLATE_LANGUAGES are stripped — job still created."""
        with patch("app._run_job"):
            data = {
                "video": (io.BytesIO(b"fake"), "clip.mp4"),
                "model": "base",
                "language": "auto",
                "translate_to": "klingon",
            }
            r = client.post("/transcribe", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        assert "job_id" in json.loads(r.data)

    def test_valid_request_returns_job_id(self, client):
        with patch("app._run_job"):
            data = {
                "video": (io.BytesIO(b"data"), "clip.mp4"),
                "model": "base",
                "language": "auto",
            }
            r = client.post("/transcribe", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        body = json.loads(r.data)
        assert "job_id" in body
        # UUID format
        uuid.UUID(body["job_id"])


# Route: GET /result/<job_id>

class TestResultRoute:
    def test_unknown_job(self, client):
        r = client.get("/result/nonexistent")
        assert r.status_code == 404

    def test_job_not_ready(self, client):
        jid = _make_job(result=None, error=None)
        r = client.get(f"/result/{jid}")
        assert r.status_code == 202

    def test_job_error(self, client):
        jid = _make_job(result=None, error="Something went wrong")
        r = client.get(f"/result/{jid}")
        assert r.status_code == 500
        assert b"Something went wrong" in r.data

    def test_job_ready(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/result/{jid}")
        assert r.status_code == 200
        body = json.loads(r.data)
        assert body["detected_language"] == "en"
        assert "langs" in body


# Route: GET /stream/<job_id>

class TestStreamRoute:
    def test_unknown_job(self, client):
        r = client.get("/stream/nonexistent")
        assert r.status_code == 404

    def test_streams_events_then_closes(self, client):
        jid = _make_job()
        q = app._jobs[jid]["queue"]
        q.put("event: progress\ndata: {\"pct\":10}\n\n")
        q.put(None)  # sentinel
        r = client.get(f"/stream/{jid}")
        assert r.status_code == 200
        assert r.content_type.startswith("text/event-stream")


# Route: GET /download/<job_id>/<lang>/<fmt>

class TestDownloadRoute:
    def test_unknown_job(self, client):
        r = client.get("/download/bad-id/en/srt")
        assert r.status_code == 404

    def test_invalid_format(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/download/{jid}/en/pdf")
        assert r.status_code == 400

    def test_language_not_available(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/download/{jid}/fr/srt")
        assert r.status_code == 404

    def test_srt_download(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/download/{jid}/en/srt")
        assert r.status_code == 200
        assert "subtitles_en.srt" in r.headers["Content-Disposition"]

    def test_vtt_download(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/download/{jid}/en/vtt")
        assert r.status_code == 200

    def test_txt_download(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/download/{jid}/en/txt")
        assert r.status_code == 200
        assert b"Hello world" in r.data

    def test_no_temp_file_leak(self, client):
        """Download must not create any files in the OS temp directory."""
        import tempfile
        before = set(Path(tempfile.gettempdir()).iterdir())
        jid = _make_job(result=_SAMPLE_RESULT)
        client.get(f"/download/{jid}/en/srt")
        after = set(Path(tempfile.gettempdir()).iterdir())
        new_files = after - before
        assert new_files == set(), f"Temp files leaked: {new_files}"


# Route: POST /dub/<job_id>/<lang>

class TestStartDubRoute:
    def test_unknown_job(self, client):
        r = client.post("/dub/nonexistent/en")
        assert r.status_code == 404

    def test_job_not_ready(self, client):
        jid = _make_job(result=None)
        r = client.post(f"/dub/{jid}/en")
        assert r.status_code == 404

    def test_language_not_in_result(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.post(f"/dub/{jid}/fr")
        assert r.status_code == 400

    def test_starts_dub_job(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        with patch("app._dub_job"):
            r = client.post(f"/dub/{jid}/en")
        assert r.status_code == 200
        body = json.loads(r.data)
        assert "dub_id" in body
        uuid.UUID(body["dub_id"])

    def test_get_not_allowed(self, client):
        jid = _make_job(result=_SAMPLE_RESULT)
        r = client.get(f"/dub/{jid}/en")
        assert r.status_code == 405


# Route: GET /stream_dub/<dub_id>

class TestStreamDubRoute:
    def test_unknown_dub(self, client):
        r = client.get("/stream_dub/nonexistent")
        assert r.status_code == 404

    def test_streams_events(self, client):
        did = _make_dub()
        q = app._dub_jobs[did]["queue"]
        q.put("event: done\ndata: {}\n\n")
        q.put(None)
        r = client.get(f"/stream_dub/{did}")
        assert r.status_code == 200
        assert r.content_type.startswith("text/event-stream")


# Route: GET /download_dub/<dub_id>

class TestDownloadDubRoute:
    def test_unknown_dub(self, client):
        r = client.get("/download_dub/nonexistent")
        assert r.status_code == 404

    def test_dub_errored(self, client):
        did = _make_dub(error="TTS failed")
        r = client.get(f"/download_dub/{did}")
        assert r.status_code == 500
        assert b"TTS failed" in r.data

    def test_dub_not_ready(self, client):
        did = _make_dub(result_path=None)
        r = client.get(f"/download_dub/{did}")
        assert r.status_code == 202

    def test_dub_ready(self, client, tmp_path):
        mp3 = tmp_path / "out.mp3"
        mp3.write_bytes(b"\xff\xfb" + b"\x00" * 128)  # minimal MP3-like bytes
        did = _make_dub(result_path=str(mp3))
        r = client.get(f"/download_dub/{did}")
        assert r.status_code == 200
        assert b"dubbed_audio.mp3" in r.headers["Content-Disposition"].encode()


# Unit tests — _ensure_translation (argostranslate interaction)

class TestEnsureTranslation:
    def _make_installed_lang(self, from_code: str, to_code: str):
        """Build minimal argostranslate language/translation stubs."""
        to_lang = MagicMock()
        to_lang.code = to_code
        translation = MagicMock()
        translation.to_lang = to_lang
        lang = MagicMock()
        lang.code = from_code
        lang.translations_to = [translation]
        return lang

    def test_already_installed_does_not_download(self):
        lang = self._make_installed_lang("en", "fr")
        import argostranslate.translate as at_translate
        import argostranslate.package as at_package
        at_translate.get_installed_languages = MagicMock(return_value=[lang])

        app._ensure_translation("en", "fr")

        at_package.update_package_index.assert_not_called()

    def test_not_installed_triggers_download(self):
        import argostranslate.translate as at_translate
        import argostranslate.package as at_package

        at_translate.get_installed_languages = MagicMock(return_value=[])
        app._pkg_index_fetched = False

        pkg = MagicMock()
        pkg.from_code = "en"
        pkg.to_code = "de"
        pkg.download.return_value = "/tmp/fake.argosmodel"
        at_package.get_available_packages = MagicMock(return_value=[pkg])
        at_package.update_package_index = MagicMock()
        at_package.install_from_path = MagicMock()

        app._ensure_translation("en", "de")

        at_package.update_package_index.assert_called_once()
        at_package.install_from_path.assert_called_once_with("/tmp/fake.argosmodel")

    def test_no_package_available_raises(self):
        import argostranslate.translate as at_translate
        import argostranslate.package as at_package

        at_translate.get_installed_languages = MagicMock(return_value=[])
        app._pkg_index_fetched = False
        at_package.get_available_packages = MagicMock(return_value=[])
        at_package.update_package_index = MagicMock()

        with pytest.raises(RuntimeError, match="No offline translation package"):
            app._ensure_translation("en", "xx")


# Unit tests — _run_job cleanup behaviour

class TestRunJobCleanup:
    def test_video_file_kept_on_success(self, tmp_path):
        """Video is intentionally kept after successful transcription (for muxing)."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        jid = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": None, "error": None}

        with patch("app._transcribe", return_value=([(0.0, 1.0, "hi")], "en")), \
             patch("app._safe_diarize", return_value=({}, [])), \
             patch("app._detect_source_bitrate", return_value="192k"), \
             patch("app.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0)
            t = threading.Thread(
                target=app._run_job,
                args=(jid, str(video), "base", "auto", []),
            )
            t.start()
            t.join(timeout=10)

        # Video is kept on success so the user can mux the dubbed audio into it
        assert video.exists(), "Input video file should be kept after successful job (needed for muxing)"
        assert app._jobs[jid]["video_path"] == str(video)

    def test_video_file_deleted_on_failure(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        jid = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": None, "error": None}

        with patch("app.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1, stderr=b"ffmpeg error")
            t = threading.Thread(
                target=app._run_job,
                args=(jid, str(video), "base", "auto", []),
            )
            t.start()
            t.join(timeout=15)

        assert not video.exists(), "Input video file should be deleted even on failure"
        assert app._jobs[jid]["error"] is not None

    def test_audio_file_deleted_on_failure(self, tmp_path):
        """If the job fails, the extracted WAV should NOT persist."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        jid = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": None, "error": None}

        with patch("app.subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=1, stderr=b"bad input")
            t = threading.Thread(
                target=app._run_job,
                args=(jid, str(video), "base", "auto", []),
            )
            t.start()
            t.join(timeout=15)

        audio_path = app._jobs[jid].get("audio_path")
        # audio_path is not set on failure (job["audio_path"] is never assigned)
        assert audio_path is None or not os.path.exists(audio_path)


# New feature routes — speakers, media_info, voices, mux

class TestSpeakersRoute:
    def test_unknown_job_returns_404(self, client):
        resp = client.get("/speakers/nonexistent")
        assert resp.status_code == 404

    def test_not_ready_returns_202(self, client):
        jid = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": None, "error": None}
        resp = client.get(f"/speakers/{jid}")
        assert resp.status_code == 202

    def test_returns_speakers_when_ready(self, client):
        jid = str(uuid.uuid4())
        app._jobs[jid] = {
            "queue": queue.Queue(),
            "result": {"langs": {}, "segments": {}},
            "error": None,
            "speakers": {"SPEAKER_00": {"gender_hint": "M", "segment_count": 5,
                                        "total_duration": 10.0, "median_f0": 120.0}},
        }
        resp = client.get(f"/speakers/{jid}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "speakers" in data
        assert "SPEAKER_00" in data["speakers"]
        assert data["diarization_available"] is True


class TestMediaInfoRoute:
    def test_unknown_job_returns_404(self, client):
        resp = client.get("/media_info/nonexistent")
        assert resp.status_code == 404

    def test_no_video_path_returns_404(self, client):
        jid = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": None, "error": None}
        resp = client.get(f"/media_info/{jid}")
        assert resp.status_code == 404

    def test_runs_ffprobe_on_valid_job(self, client, tmp_path):
        video = tmp_path / "sample.mp4"
        video.write_bytes(b"fake")
        jid = str(uuid.uuid4())
        app._jobs[jid] = {
            "queue": queue.Queue(), "result": {}, "error": None,
            "video_path": str(video),
        }
        fake_probe = {"format": {"duration": "10.0", "size": "102400"}, "streams": []}
        with patch("app._run_ffprobe", return_value=fake_probe):
            resp = client.get(f"/media_info/{jid}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "format" in data


class TestMuxRoute:
    def test_unknown_job_returns_404(self, client):
        resp = client.post("/mux/nojob/nodub", json={"mode": "replace"})
        assert resp.status_code == 404

    def test_dub_not_complete_returns_400(self, client):
        jid = str(uuid.uuid4())
        did = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": {}, "error": None,
                          "video_path": "/some/path.mp4"}
        app._dub_jobs[did] = {"queue": queue.Queue(), "result_path": None, "error": None}
        resp = client.post(f"/mux/{jid}/{did}", json={"mode": "replace"})
        assert resp.status_code == 400

    def test_invalid_mode_returns_400(self, client, tmp_path):
        """If neither audio track is selected, the mux must be rejected."""
        video = tmp_path / "v.mp4"
        video.write_bytes(b"v")
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"a")
        jid = str(uuid.uuid4())
        did = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": {}, "error": None,
                          "video_path": str(video)}
        app._dub_jobs[did] = {"queue": queue.Queue(), "result_path": str(audio), "error": None}
        resp = client.post(f"/mux/{jid}/{did}",
                           json={"include_original": False, "include_dubbed": False})
        assert resp.status_code == 400

    def test_valid_mux_starts_job(self, client, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"v")
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"a")
        jid = str(uuid.uuid4())
        did = str(uuid.uuid4())
        app._jobs[jid] = {"queue": queue.Queue(), "result": {}, "error": None,
                          "video_path": str(video)}
        app._dub_jobs[did] = {"queue": queue.Queue(), "result_path": str(audio), "error": None}
        with patch("threading.Thread"):
            resp = client.post(f"/mux/{jid}/{did}",
                               json={"include_original": True, "include_dubbed": True,
                                     "output_format": "mp4"})
        assert resp.status_code == 200
        assert "mux_id" in resp.get_json()


class TestDubRouteVoiceProfiles:
    def test_dub_accepts_voice_profiles_json(self, client):
        jid = str(uuid.uuid4())
        app._jobs[jid] = {
            "queue": queue.Queue(),
            "result": {"langs": {"en": {}}, "segments": {"en": [(0, 1, "hi")]}},
            "error": None,
        }
        body = {"voice_profiles": {"SPEAKER_00": {"voice": "en-US-GuyNeural",
                                                   "rate": "+2%", "pitch": "+1Hz"}}}
        with patch("threading.Thread"):
            resp = client.post(f"/dub/{jid}/en",
                               data=json.dumps(body),
                               content_type="application/json")
        assert resp.status_code == 200
        assert "dub_id" in resp.get_json()


class TestDiarize:
    def test_fallback_without_librosa(self):
        """_diarize falls back gracefully if sklearn/librosa are unavailable."""
        segs = [(0.0, 1.0, "hello"), (1.0, 2.0, "world")]
        with patch.dict("sys.modules", {"librosa": None, "sklearn": None,
                                        "sklearn.cluster": None,
                                        "sklearn.preprocessing": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                speakers, spk_segs = app._diarize("fake.wav", segs)
        assert "SPEAKER_00" in speakers
        assert len(spk_segs) == len(segs)

    def test_build_speaker_dicts(self):
        """_build_speaker_dicts returns valid speaker and seg structures."""
        import numpy as np
        segs = [(0.0, 1.5, "a"), (1.5, 3.0, "b"), (3.0, 4.5, "c")]
        labels = np.array([0, 1, 0])
        f0 = [120.0, 220.0, 115.0]
        speakers, spk_segs = app._build_speaker_dicts(segs, labels, f0)
        assert "SPEAKER_00" in speakers
        assert "SPEAKER_01" in speakers
        assert speakers["SPEAKER_00"]["gender_hint"] == "M"
        assert speakers["SPEAKER_01"]["gender_hint"] == "F"
        assert len(spk_segs) == 3


# New helpers from the v1.2.0 refactor

class TestErrorHelper:
    def test_default_code(self):
        with app.app.app_context():
            resp, code = app._error("oops")
        assert code == 400
        assert resp.get_json() == {"error": "oops"}

    def test_custom_code(self):
        with app.app.app_context():
            _, code = app._error("nope", 404)
        assert code == 404


class TestWhisperModelCache:
    def test_caches_per_size(self):
        app._whisper_cache.clear()
        with patch("app.WhisperModel") as MockWM:
            MockWM.side_effect = lambda *a, **kw: MagicMock(name=f"M-{a[0]}")
            m1 = app._get_whisper_model("base")
            m2 = app._get_whisper_model("base")
            m3 = app._get_whisper_model("small")
        assert m1 is m2
        assert m1 is not m3
        assert MockWM.call_count == 2  # base + small, not base twice
        app._whisper_cache.clear()


class TestCleanupOnFailure:
    def test_deletes_existing_files(self, tmp_path):
        a = tmp_path / "a.wav"; a.write_bytes(b"x")
        b = tmp_path / "b.mp4"; b.write_bytes(b"y")
        app._cleanup_on_failure(str(a), str(b))
        assert not a.exists() and not b.exists()

    def test_tolerates_missing(self, tmp_path):
        # Must not raise on non-existent paths
        app._cleanup_on_failure(str(tmp_path / "nope.wav"), "")


class TestSafeDiarize:
    def test_returns_fallback_on_exception(self):
        with patch("app._diarize", side_effect=RuntimeError("boom")):
            speakers, _ = app._safe_diarize("x.wav", [(0.0, 1.0, "hi")])
        assert "SPEAKER_00" in speakers

    def test_passes_through_on_success(self):
        fake = ({"SPEAKER_00": {"gender_hint": "M"}}, [(0.0, 1.0, "hi", "SPEAKER_00")])
        with patch("app._diarize", return_value=fake):
            speakers, _ = app._safe_diarize("x.wav", [(0.0, 1.0, "hi")])
        assert speakers["SPEAKER_00"]["gender_hint"] == "M"


class TestJobsLockExists:
    def test_locks_present(self):
        assert isinstance(app._jobs_lock, type(threading.Lock()))
        assert isinstance(app._whisper_lock, type(threading.Lock()))


class TestConfigPathSafety:
    def test_realpath_resolves_traversal(self, client, tmp_path):
        # Pass a path with .. — set_config should realpath() it
        nested = tmp_path / "sub" / ".." / "real"
        (tmp_path / "real").mkdir()
        resp = client.post("/config",
                           data=json.dumps({"output_folder": str(nested)}),
                           content_type="application/json")
        assert resp.status_code == 200
        body = resp.get_json()
        # Resolved path should not contain ".."
        assert ".." not in body.get("output_folder", "")


class TestStudio:
    def test_classify_upload(self):
        assert app._classify_studio_upload(".mp4") == "video"
        assert app._classify_studio_upload(".mp3") == "audio"
        assert app._classify_studio_upload(".srt") == "subtitle"
        assert app._classify_studio_upload(".xyz") == ""

    def test_upload_rejects_unsupported(self, client):
        data = {"file": (io.BytesIO(b"x"), "thing.xyz")}
        resp = client.post("/studio/upload", data=data,
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Unsupported" in resp.get_json()["error"]

    def test_upload_rejects_no_file(self, client):
        resp = client.post("/studio/upload", data={},
                           content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_build_requires_video(self, client):
        resp = client.post("/studio/build",
                           data=json.dumps({"audio_tracks": [], "subtitle_tracks": []}),
                           content_type="application/json")
        assert resp.status_code == 400
        assert "video" in resp.get_json()["error"].lower()

    def test_validate_studio_request_raises(self):
        with pytest.raises(app._StudioValidationError):
            app._validate_studio_request({})
