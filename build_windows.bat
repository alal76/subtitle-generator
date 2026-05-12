@echo off
setlocal EnableDelayedExpansion
title Video Subtitler — Windows Build

echo.
echo ============================================================
echo   Video Subtitler — Windows Build Script
echo ============================================================
echo.

:: ── working directory = script location ─────────────────────────────────────
cd /d "%~dp0"

:: ============================================================
:: 1. Check for winget (Windows Package Manager)
:: ============================================================
where winget >nul 2>&1
if errorlevel 1 (
    echo [ERROR] winget not found.
    echo         Install "App Installer" from the Microsoft Store, then re-run.
    pause
    exit /b 1
)

:: ============================================================
:: 2. Install / verify Python 3.11
:: ============================================================
echo [1/6] Checking Python...
where python >nul 2>&1
if errorlevel 1 (
    echo       Python not found — installing via winget...
    winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Python installation failed. Install manually from https://python.org
        pause & exit /b 1
    )
    :: Refresh PATH so python is visible in this session
    for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON_EXE=%%i
) else (
    for /f "tokens=*" %%i in ('where python') do set PYTHON_EXE=%%i
)

:: Validate version (need 3.9+)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo       Found Python !PY_VER! at !PYTHON_EXE!

:: ============================================================
:: 3. Install / verify ffmpeg
:: ============================================================
echo [2/6] Checking ffmpeg...
if exist "ffmpeg.exe" (
    echo       Found bundled ffmpeg.exe in project folder.
    set FFMPEG_READY=1
) else (
    where ffmpeg >nul 2>&1
    if not errorlevel 1 (
        echo       Found ffmpeg on PATH — copying to project folder for bundling...
        for /f "tokens=*" %%i in ('where ffmpeg') do copy "%%i" "ffmpeg.exe" >nul
        set FFMPEG_READY=1
    ) else (
        echo       ffmpeg not found — installing via winget...
        winget install --id Gyan.FFmpeg --silent --accept-package-agreements --accept-source-agreements
        if errorlevel 1 (
            echo [WARN] winget ffmpeg install failed. Trying direct download...
            call :download_ffmpeg
        ) else (
            :: Refresh PATH
            for /f "tokens=*" %%i in ('where ffmpeg 2^>nul') do (
                copy "%%i" "ffmpeg.exe" >nul
                set FFMPEG_READY=1
            )
        )
    )
)

if not defined FFMPEG_READY (
    echo [ERROR] Could not locate or install ffmpeg.
    echo         Download manually from https://ffmpeg.org/download.html
    echo         Place ffmpeg.exe in: %CD%
    pause & exit /b 1
)

:: ============================================================
:: 4. Create virtual environment
:: ============================================================
echo [3/6] Setting up virtual environment...
if not exist ".venv\Scripts\activate.bat" (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo       Created .venv
) else (
    echo       Existing .venv found — reusing.
)

call .venv\Scripts\activate.bat

:: ============================================================
:: 5. Install Python dependencies + PyInstaller
:: ============================================================
echo [4/6] Installing Python packages...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause & exit /b 1
)
python -m pip install --quiet pyinstaller
if errorlevel 1 (
    echo [ERROR] PyInstaller install failed.
    pause & exit /b 1
)
echo       All packages installed.

:: ============================================================
:: 6. Build the executable
:: ============================================================
echo [5/6] Building executable with PyInstaller...
if exist "build" rmdir /s /q "build"
if exist "dist"  rmdir /s /q "dist"

python -m PyInstaller --clean --noconfirm video_subtitler.spec
if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed — check output above.
    pause & exit /b 1
)

:: ============================================================
:: 7. Verify output
:: ============================================================
echo [6/6] Verifying output...
set EXE=dist\VideoSubtitler\VideoSubtitler.exe
if not exist "%EXE%" (
    echo [ERROR] Expected executable not found: %EXE%
    pause & exit /b 1
)

for %%A in ("%EXE%") do set EXE_SIZE=%%~zA
set /a EXE_MB=!EXE_SIZE! / 1048576

echo.
echo ============================================================
echo   BUILD SUCCESSFUL
echo ============================================================
echo   Executable : %CD%\%EXE%
echo.
echo   To run     : double-click VideoSubtitler.exe
echo                (or run it from this terminal)
echo.
echo   To share   : zip the entire dist\VideoSubtitler\ folder.
echo                Recipients do NOT need Python or ffmpeg installed.
echo ============================================================
echo.

:: Optional: open the dist folder in Explorer
set /p OPEN_DIST="Open dist folder in Explorer? [Y/n]: "
if /i not "!OPEN_DIST!"=="n" explorer "%CD%\dist\VideoSubtitler"

pause
exit /b 0


:: ============================================================
:: Helper: download ffmpeg via PowerShell as a fallback
:: ============================================================
:download_ffmpeg
echo       Attempting PowerShell download of ffmpeg...
set FFMPEG_ZIP=ffmpeg_download.zip
set FFMPEG_URL=https://github.com/GyanD/codexffmpeg/releases/download/7.1.1/ffmpeg-7.1.1-essentials_build.zip

powershell -NoProfile -Command ^
  "[Net.ServicePointManager]::SecurityProtocol='Tls12';" ^
  "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing"

if not exist "%FFMPEG_ZIP%" (
    echo [WARN] Download failed.
    goto :eof
)

echo       Extracting ffmpeg.exe...
powershell -NoProfile -Command ^
  "Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath 'ffmpeg_tmp' -Force"

:: The zip has a versioned subdirectory — find ffmpeg.exe recursively
for /r "ffmpeg_tmp" %%f in (ffmpeg.exe) do (
    copy "%%f" "ffmpeg.exe" >nul
    set FFMPEG_READY=1
)

rmdir /s /q "ffmpeg_tmp" 2>nul
del /q "%FFMPEG_ZIP%"      2>nul

if defined FFMPEG_READY (
    echo       ffmpeg.exe downloaded and placed in project folder.
) else (
    echo [WARN] Could not extract ffmpeg.exe from archive.
)
goto :eof
