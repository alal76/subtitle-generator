#!/usr/bin/env python3
"""
build.py — Build VideoSubtitler into a standalone executable.

  macOS:   python build.py          → dist/VideoSubtitler.app
  Windows: python build.py          → dist/VideoSubtitler/VideoSubtitler.exe
  Linux:   python build.py          → dist/VideoSubtitler/VideoSubtitler

Optional flags:
  --version X.Y.Z   Override the build version string (default: reads from spec)
  --skip-clean      Keep previous build/ and dist/ directories
  --checksum        Write a SHA-256 checksum file alongside the output

Requirements:
  - Run from inside the project virtualenv (or ensure pyinstaller is installed)
  - macOS/Linux: brew install ffmpeg  or  sudo apt install ffmpeg
  - Windows:     place ffmpeg.exe + ffprobe.exe in the project folder, or
                 run build_windows.bat which handles everything automatically
"""

import argparse
import hashlib
import os
import subprocess
import sys
import shutil
from pathlib import Path

APP_VERSION = "1.2.0"  # bump on each release


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build VideoSubtitler")
    parser.add_argument("--version", default=APP_VERSION, help="Version string to embed")
    parser.add_argument("--skip-clean", action="store_true", help="Keep previous build artefacts")
    parser.add_argument("--checksum", action="store_true", help="Write SHA-256 checksum file")
    args = parser.parse_args()

    here = Path(__file__).parent.resolve()
    os.chdir(here)

    # Ensure PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found — installing…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Clean previous build artefacts
    if not args.skip_clean:
        for d in ("build", "dist"):
            target = here / d
            if target.exists():
                print(f"Removing old {d}/…")
                shutil.rmtree(target)

    # Patch version into the spec's info_plist (macOS) via env var
    env = os.environ.copy()
    env["VS_VERSION"] = args.version

    print(f"\nBuilding version {args.version} for {sys.platform}…")
    result = subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            "video_subtitler.spec",
        ],
        cwd=here,
        env=env,
    )

    if result.returncode != 0:
        print("\n[ERROR] Build failed — see output above.")
        sys.exit(1)

    dist = here / "dist"
    if sys.platform == "darwin":
        out = dist / "VideoSubtitler.app"
        exe = out / "Contents" / "MacOS" / "VideoSubtitler"
        print(f"\n✓  macOS app bundle: {out}")
        print("   To run:        open dist/VideoSubtitler.app")
        print("   To distribute: zip or use create-dmg to make a .dmg")
    elif sys.platform == "win32":
        out = dist / "VideoSubtitler" / "VideoSubtitler.exe"
        exe = out
        print(f"\n✓  Windows executable: {out}")
        print("   Distribute the entire dist/VideoSubtitler/ folder (zip it).")
    else:
        out = dist / "VideoSubtitler" / "VideoSubtitler"
        exe = out
        print(f"\n✓  Linux executable: {out}")
        print("   Distribute the entire dist/VideoSubtitler/ folder (tar it).")

    if exe.exists():
        size_mb = exe.stat().st_size / 1_048_576
        print(f"   Size:           {size_mb:.1f} MB")

        if args.checksum:
            digest = sha256_of(exe)
            cs_path = exe.parent / f"{exe.name}.sha256"
            cs_path.write_text(f"{digest}  {exe.name}\n")
            print(f"   SHA-256 file:   {cs_path}")

    print(f"\n   Version:        {args.version}")


if __name__ == "__main__":
    main()
