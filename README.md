# Jarvis — Hand-Tracked AI Desktop Controller

Control your PC with hand gestures and summon an AI agent with your voice. No mouse required.

## What it does

- **Tracks your hand** via webcam in real time (GPU-accelerated C++ DLL or CPU MediaPipe fallback)
- **Maps gestures to mouse actions** — move, left click, right click, scroll
- **Fist gesture** held for 0.5s activates voice input
- **Whisper** transcribes what you say
- **Llama 3.1** (via NVIDIA NIM) interprets the command and picks the right tool
- **Jarvis speaks back** in a British male voice (`en-GB-RyanNeural`)
- **Persistent memory** — Jarvis saves what it learns across sessions in `lessons.json`

## Gestures

| Gesture | Action |
|---|---|
| Open hand, move | Move cursor (tracks palm landmark) |
| Thumb + index pinch | Left click |
| Thumb + middle pinch | Right click |
| Thumb + pinky pinch | Scroll (move hand up/down) |
| Fist (hold 0.5s) | Start voice input |
| Release fist | Stop recording, process command |
| `T` key | Toggle the HUD window |
| `Esc` | Quit |

## What Jarvis can do via voice

- Open apps (Chrome, VS Code, Spotify, Discord, etc.) or any URL/domain
- Run any PowerShell command
- Type text, press keyboard shortcuts
- Click, double-click, drag, or scroll anywhere on screen
- **Read the screen** using computer vision — finds buttons, videos, and UI elements by sight, then clicks them
- Take screenshots, read/write files, get/set clipboard
- Search the web
- Speak responses aloud
- Admit when it can't do something rather than hallucinating success

## Setup

### Requirements

- Windows 10/11
- Python 3.10+ (tested on 3.14)
- A webcam
- An NVIDIA API key — free tier at [build.nvidia.com](https://build.nvidia.com)

### Quick start (recommended)

```bash
git clone https://github.com/T-Perm/Jarvis.py
cd Jarvis.py
setup.bat
```

`setup.bat` will create a venv, install dependencies, download the hand landmark model, and create a `.env` file for you. Open `.env`, paste your NVIDIA API key, then run:

```bash
venv\Scripts\activate
python app.py
```

### Manual install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** this project uses `pygame-ce` (Community Edition), not standard `pygame`. They conflict — do not install both.

Download the hand landmark model:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Copy `.env.example` to `.env` and fill in your key:

```env
NVIDIA_API_KEY=your_key_here
NIM_MODEL=meta/llama-3.1-8b-instruct
```

## Architecture

```
app.py          — main loop: gesture tracking, mouse control, HUD, fist trigger
jarvis.py       — voice pipeline: STT → LLM → tool dispatch → TTS
hand_tracker.cc — C++ source for the GPU DLL backend (optional, faster)
```

**Tracker backends** (auto-selected at startup):
- `hand_tracker.dll` — compiled C++ with OpenCV, GPU-capable, preferred
- Python MediaPipe — CPU fallback, no DLL needed

**Voice pipeline:**
1. `sounddevice` records mic input while fist is held
2. `faster-whisper` (`base.en`, CPU int8) transcribes the audio
3. NVIDIA NIM (OpenAI-compatible API) runs Llama 3.1 with tool use
4. Tools are dispatched locally (PowerShell, pyautogui, webbrowser, etc.)
5. `edge-tts` synthesizes the response as `en-GB-RyanNeural` MP3, played via `pygame-ce`

**Screen vision:**
The `read_screen` tool takes a screenshot and sends it to `nvidia/llama-3.2-11b-vision-instruct` to describe what's on screen with pixel positions. Jarvis then uses `click_screen` with the coordinates it finds.

## Notes

- The DLL backend requires `opencv_world3416.dll` alongside `hand_tracker.dll`. If missing, the Python MediaPipe fallback runs automatically.
- `hand_landmarker.task` and `*.dll` files are gitignored — `setup.bat` handles the model download automatically.
- Swap the LLM via the `NIM_MODEL` env var to any model on NVIDIA NIM.
