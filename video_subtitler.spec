# PyInstaller spec file for Video Subtitler
# Works on macOS, Windows, and Linux — run from the project root with:
#   pyinstaller video_subtitler.spec
# Or use the build script:  python build.py

import sys
import os
from pathlib import Path

IS_WIN   = sys.platform == "win32"
IS_MAC   = sys.platform == "darwin"
IS_LIN   = sys.platform.startswith("linux")
VERSION  = os.environ.get("VS_VERSION", "1.2.0")
VENV     = Path(".venv")
SP       = VENV / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

# ── locate ffmpeg binary ────────────────────────────────────────────────────
def find_ffmpeg():
    """Search common locations for ffmpeg binary to bundle."""
    candidates = [
        # macOS Homebrew (Apple Silicon / Intel)
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        # Linux (most distros)
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/snap/bin/ffmpeg",
    ]
    if IS_WIN:
        # Windows: project folder first (placed by build_windows.bat), then common installs
        candidates = [
            str(Path(__file__).parent / "ffmpeg.exe"),          # copied by build_windows.bat
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\WinGet\Packages\ffmpeg.exe"),
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "ffmpeg not found. On macOS run: brew install ffmpeg\n"
        "On Windows download from https://ffmpeg.org/download.html and place "
        "ffmpeg.exe next to this spec file."
    )

FFMPEG_SRC = find_ffmpeg()
FFMPEG_DST = "ffmpeg.exe" if IS_WIN else "ffmpeg"

def find_ffprobe():
    """Locate ffprobe alongside ffmpeg."""
    ffmpeg_dir = os.path.dirname(FFMPEG_SRC)
    name = "ffprobe.exe" if IS_WIN else "ffprobe"
    p = os.path.join(ffmpeg_dir, name)
    if os.path.isfile(p):
        return p
    return None  # optional — runtime falls back to PATH

FFPROBE_SRC = find_ffprobe()

# ── data files ──────────────────────────────────────────────────────────────
datas = [
    (FFMPEG_SRC, "."),                                            # ffmpeg binary
    *([(FFPROBE_SRC, ".")] if FFPROBE_SRC else []),               # ffprobe (if found)
    (str(SP / "faster_whisper"),    "faster_whisper"),
    (str(SP / "ctranslate2"),       "ctranslate2"),
    (str(SP / "argostranslate"),    "argostranslate"),
    (str(SP / "flask"),             "flask"),
    (str(SP / "werkzeug"),          "werkzeug"),
    (str(SP / "jinja2"),            "jinja2"),
    (str(SP / "click"),             "click"),
    (str(SP / "itsdangerous"),      "itsdangerous"),
    (str(SP / "markupsafe"),        "markupsafe"),
    (str(SP / "edge_tts"),          "edge_tts"),
    (str(SP / "pydub"),             "pydub"),
    (str(SP / "librosa"),           "librosa"),
    (str(SP / "sklearn"),           "sklearn"),
]

# ── hidden imports ───────────────────────────────────────────────────────────
hidden = [
    "flask",
    "werkzeug",
    "jinja2",
    "click",
    "faster_whisper",
    "ctranslate2",
    "argostranslate",
    "argostranslate.package",
    "argostranslate.translate",
    "huggingface_hub",
    "tokenizers",
    "numpy",
    "av",
    "edge_tts",
    "pydub",
    "aiohttp",
    "aiofiles",
    "librosa",
    "sklearn",
    "sklearn.cluster",
    "sklearn.preprocessing",
    "sklearn.metrics",        # silhouette_score for speaker count detection
    "scipy",
    "soundfile",
]

# ── analysis ─────────────────────────────────────────────────────────────────
a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "PyQt5", "PyQt6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VideoSubtitler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # no terminal window; change to True for debugging
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VideoSubtitler",
)

# macOS: wrap in a .app bundle
if IS_MAC:
    app = BUNDLE(
        coll,
        name="VideoSubtitler.app",
        bundle_identifier="com.videosubtitler.app",
        info_plist={
            "CFBundleDisplayName":        "Video Subtitler",
            "CFBundleVersion":            VERSION,
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable":    True,
        },
    )
