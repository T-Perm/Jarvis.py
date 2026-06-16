@echo off
echo === Jarvis Setup ===
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://python.org and re-run this script.
    pause
    exit /b 1
)

python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

if not exist hand_landmarker.task (
    echo.
    echo Downloading hand landmark model...
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
)

if not exist .env (
    copy .env.example .env >nul
    echo.
    echo .env created -- open it and paste your NVIDIA API key before running.
    echo Get a free key at https://build.nvidia.com
    echo.
)

echo.
echo Setup complete. Run:  venv\Scripts\activate ^&^& python app.py
pause
