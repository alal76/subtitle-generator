#!/usr/bin/env python3
"""
build.py — Build VideoSubtitler into a standalone executable.

macOS:  python build.py          → dist/VideoSubtitler.app
Windows: python build.py         → dist/VideoSubtitler/VideoSubtitler.exe

Requirements:
  - Run from inside the project virtualenv (or ensure pyinstaller is installed)
  - macOS:   brew install ffmpeg  (already installed if you used run.sh)
  - Windows: place ffmpeg.exe in the project folder, or install via
             winget install Gyan.FFmpeg  (then update PATH / spec candidates)
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path


def main():
    here = Path(__file__).parent.resolve()
    os.chdir(here)

    # Ensure PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found — installing…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Clean previous build artefacts
    for d in ("build", "dist"):
        target = here / d
        if target.exists():
            print(f"Removing old {d}/…")
            shutil.rmtree(target)

    print(f"\nBuilding for {sys.platform}…")
    result = subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "video_subtitler.spec",
        ],
        cwd=here,
    )

    if result.returncode != 0:
        print("\n[ERROR] Build failed — see output above.")
        sys.exit(1)

    dist = here / "dist"
    if sys.platform == "darwin":
        out = dist / "VideoSubtitler.app"
        print(f"\n✓  macOS app bundle: {out}")
        print("   To run: open dist/VideoSubtitler.app")
        print("   To distribute: zip or use create-dmg to make a .dmg")
    else:
        out = dist / "VideoSubtitler" / "VideoSubtitler.exe"
        print(f"\n✓  Windows executable: {out}")
        print("   Distribute the entire dist/VideoSubtitler/ folder (zip it).")


if __name__ == "__main__":
    main()
