#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
  echo "ffmpeg not found. Installing via Homebrew..."
  brew install ffmpeg
fi

# Create venv if needed
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# Install / upgrade deps
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "Starting Video Subtitler..."
python app.py
