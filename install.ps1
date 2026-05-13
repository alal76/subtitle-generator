# install.ps1 — One-command install + launch for Video Subtitler (Windows)
#
# Usage (PowerShell, run as your normal user):
#   irm https://raw.githubusercontent.com/alal76/subtitle-generator/main/install.ps1 | iex
#
# What it does:
#   1. Installs Git, Python 3.11, and ffmpeg via winget (if not already present)
#   2. Clones the repo (or pulls latest if already cloned)
#   3. Creates a Python virtual environment
#   4. Installs all Python dependencies
#   5. Launches the app at http://127.0.0.1:5001

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REPO = "https://github.com/alal76/subtitle-generator.git"
$DIR  = if ($env:VIDEO_SUBTITLER_DIR) { $env:VIDEO_SUBTITLER_DIR } else { Join-Path $HOME "subtitle-generator" }
$PORT = if ($env:PORT) { $env:PORT } else { "5001" }

function Info  ($msg) { Write-Host "[info]  $msg" -ForegroundColor Cyan }
function Ok    ($msg) { Write-Host "[ ok ]  $msg" -ForegroundColor Green }
function Warn  ($msg) { Write-Host "[warn]  $msg" -ForegroundColor Yellow }
function Fail  ($msg) { Write-Host "[fail]  $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  ╔═══════════════════════════════════════════╗"
Write-Host "  ║     Video Subtitler — Install & Launch    ║"
Write-Host "  ╚═══════════════════════════════════════════╝"
Write-Host ""

# ── helper: run winget silently ───────────────────────────────────────────────
function Install-WinGet ($id, $label) {
    Info "Installing $label via winget…"
    winget install --id $id -e --source winget --accept-package-agreements --accept-source-agreements
    # Refresh PATH so the newly installed tool is visible
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

# ── 1. Prerequisite tools ─────────────────────────────────────────────────────
if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
    Fail "winget (App Installer) is required. Update Windows or install it from the Microsoft Store."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Install-WinGet "Git.Git" "Git"
}
Ok "git $(git --version)"

# Prefer python3, fall back to python
$PY = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { $null }

if (-not $PY) {
    Install-WinGet "Python.Python.3.11" "Python 3.11"
    $PY = "python"
}

$pyVer = & $PY -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>&1
$parts = $pyVer -split "\."
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 9)) {
    Fail "Python $pyVer found — 3.9 or newer is required. Uninstall the old version and re-run."
}
Ok "Python $pyVer"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Install-WinGet "Gyan.FFmpeg" "ffmpeg"
}
$ffVer = (ffmpeg -version 2>&1 | Select-Object -First 1) -replace '.*version (\S+).*','$1'
Ok "ffmpeg $ffVer"

if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    Warn "ffprobe not found — media info and bitrate matching will be limited."
} else {
    $fpVer = (ffprobe -version 2>&1 | Select-Object -First 1) -replace '.*version (\S+).*','$1'
    Ok "ffprobe $fpVer"
}

# ── 2. Clone or update ────────────────────────────────────────────────────────
if (Test-Path (Join-Path $DIR ".git")) {
    Info "Updating existing clone at $DIR …"
    git -C $DIR pull --ff-only
    Ok "Repository up to date"
} else {
    Info "Cloning repository into $DIR …"
    git clone $REPO $DIR
    Ok "Repository cloned"
}

Set-Location $DIR

# ── 3. Virtual environment ────────────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
    Info "Creating virtual environment…"
    & $PY -m venv .venv
}
$VENV_PY = Join-Path $DIR ".venv\Scripts\python.exe"
Ok "Virtual environment ready"

# ── 4. Python dependencies ────────────────────────────────────────────────────
Info "Installing Python dependencies (may take a minute on first run)…"
& $VENV_PY -m pip install --quiet --upgrade pip
& $VENV_PY -m pip install --quiet -r requirements.txt
Ok "Dependencies installed"

# ── 5. Launch ─────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ✓ Installation complete." -ForegroundColor Green
Write-Host "  → Opening http://127.0.0.1:$PORT"
Write-Host "  → Close this window or press Ctrl-C to stop."
Write-Host ""

$env:PORT = $PORT
& $VENV_PY app.py
