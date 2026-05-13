#!/usr/bin/env bash
# install.sh — One-command install + launch for Video Subtitler (macOS & Linux)
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.sh | bash
#
# What it does:
#   1. Installs ffmpeg + ffprobe if missing (Homebrew / apt / dnf / pacman)
#   2. Clones the repo (or pulls latest if already cloned)
#   3. Creates a Python virtual environment
#   4. Installs all Python dependencies
#   5. Launches the app at http://127.0.0.1:5001

set -e

REPO="https://github.com/alal76/subtitle-generator.git"
DIR="${VIDEO_SUBTITLER_DIR:-$HOME/subtitle-generator}"
PORT="${PORT:-5001}"

# ── pretty print helpers ──────────────────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[ ok ]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()   { printf '\033[1;31m[fail]\033[0m  %s\n' "$*" >&2; exit 1; }

echo ""
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║     Video Subtitler — Install & Launch    ║"
echo "  ╚═══════════════════════════════════════════╝"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
PY="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
[[ -z "$PY" ]] && die "Python 3.9+ is required but not found. Install it first."
PY_VER="$( "$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' )"
PY_MAJ="${PY_VER%%.*}"; PY_MIN="${PY_VER#*.}"
(( PY_MAJ < 3 || (PY_MAJ == 3 && PY_MIN < 9) )) \
  && die "Python $PY_VER found — 3.9 or newer is required."
ok "Python $PY_VER"

# ── 2. ffmpeg / ffprobe ───────────────────────────────────────────────────────
install_ffmpeg() {
  if command -v brew &>/dev/null; then
    info "Installing ffmpeg via Homebrew…"
    brew install ffmpeg
  elif command -v apt-get &>/dev/null; then
    info "Installing ffmpeg via apt…"
    sudo apt-get update -qq && sudo apt-get install -y ffmpeg
  elif command -v dnf &>/dev/null; then
    info "Installing ffmpeg via dnf…"
    sudo dnf install -y ffmpeg
  elif command -v pacman &>/dev/null; then
    info "Installing ffmpeg via pacman…"
    sudo pacman -S --noconfirm ffmpeg
  elif command -v zypper &>/dev/null; then
    info "Installing ffmpeg via zypper…"
    sudo zypper install -y ffmpeg
  else
    die "ffmpeg not found and no known package manager available.\n       Install ffmpeg manually and re-run this script."
  fi
}

if command -v ffmpeg &>/dev/null; then
  ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
  install_ffmpeg
  command -v ffmpeg &>/dev/null || die "ffmpeg installation failed."
  ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
fi

if command -v ffprobe &>/dev/null; then
  ok "ffprobe $(ffprobe -version 2>&1 | head -1 | awk '{print $3}')"
else
  warn "ffprobe not found — media info and bitrate matching will be limited."
fi

# ── 3. Clone or update ────────────────────────────────────────────────────────
if [[ -d "$DIR/.git" ]]; then
  info "Updating existing clone at $DIR …"
  git -C "$DIR" pull --ff-only
  ok "Repository up to date"
else
  info "Cloning repository into $DIR …"
  git clone "$REPO" "$DIR"
  ok "Repository cloned"
fi

cd "$DIR"

# ── 4. Virtual environment ────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
  info "Creating virtual environment…"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtual environment active"

# ── 5. Python dependencies ────────────────────────────────────────────────────
info "Installing Python dependencies (this may take a minute on first run)…"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies installed"

# ── 6. Launch ─────────────────────────────────────────────────────────────────
echo ""
echo "  ✓ Installation complete."
echo "  → Opening http://127.0.0.1:$PORT"
echo "  → Press Ctrl-C to stop."
echo ""
export PORT="$PORT"
exec python app.py
