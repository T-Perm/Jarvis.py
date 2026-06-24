@echo off
REM ============================================================
REM  Build a single standalone Jarvis.exe with PyInstaller.
REM  Run setup.bat first so venv\ exists with dependencies.
REM  Output: dist\Jarvis.exe
REM ============================================================
echo === Building Jarvis.exe ===
echo.

if not exist venv\Scripts\python.exe (
    echo venv not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM PyInstaller is the only extra build-time dependency.
pip install --quiet pyinstaller

REM Bundle the Whisper speech model so the exe needs no first-run download.
REM (Skip this block to ship a smaller exe that downloads base.en on first run.)
if not exist whisper-base.en (
    echo Downloading Whisper base.en model...
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-base.en', local_dir='whisper-base.en')"
)

REM Make sure the hand-landmark model is present (read by hand_tracker.dll).
if not exist hand_landmarker.task (
    echo Downloading hand landmark model...
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
)

pyinstaller Jarvis.spec --noconfirm --distpath dist --workpath build_pyi

echo.
echo Done. Executable is at dist\Jarvis.exe
echo Put a .env file (with NVIDIA_API_KEY) next to it before running.
pause
