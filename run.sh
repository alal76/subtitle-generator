#!/usr/bin/env bash
# run.sh — Start Video Subtitler from source (macOS / Linux)
#
# Usage:
#   ./run.sh              # default port 5001
#   PORT=8080 ./run.sh    # custom port
#   ./run.sh --help       # show this message

set -e

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

APP_PORT="${PORT:-5001}"

# ── Python version check ────────────────────────────────────────────────────
PY="$(command -v python3 || command -v python || true)"
if [[ -z "$PY" ]]; then
  echo "[ERROR] Python 3.9+ is required but not found."
  exit 1
fi
PY_VER="$( "$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' )"
PY_MAJOR="${PY_VER%%.*}"
PY_MINOR="${PY_VER#*.}"
if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 9) )); then
  echo "[ERROR] Python $PY_VER found, but 3.9+ is required."
  exit 1
fi
echo "Python $PY_VER  (✓)"

# ── ffmpeg + ffprobe ────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
  if command -v brew &>/dev/null; then
    echo "ffmpeg not found — installing via Homebrew…"
    brew install ffmpeg
  else
    echo "[ERROR] ffmpeg not found. Install it with your package manager:"
    echo "  macOS:  brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    echo "  Fedora: sudo dnf install ffmpeg"
    exit 1
  fi
fi
echo "ffmpeg  $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')  (✓)"

if ! command -v ffprobe &>/dev/null; then
  echo "[WARN] ffprobe not found — Media Info and bitrate matching may not work."
  echo "       Install it alongside ffmpeg (usually the same package)."
else
  echo "ffprobe $(ffprobe -version 2>&1 | head -1 | awk '{print $3}')  (✓)"
fi

# ── virtual environment ─────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment…"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ── Python dependencies ─────────────────────────────────────────────────────
echo "Checking Python packages…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "Starting Video Subtitler on port $APP_PORT…"
export PORT="$APP_PORT"
exec python app.py
