# Jarvis — Hand-Tracked AI Desktop Controller

Control your PC with hand gestures and talk to an AI agent. No mouse required.

I built this because I wanted to see if I could replace my mouse entirely with a webcam. Turns out, mostly yes.

---

## How it works

You hold up your hand in front of your webcam and Jarvis tracks it. Moving your hand moves the cursor. Pinch your thumb and index finger to click. Make a fist for half a second and it starts listening to you — say what you want, release the fist, and it does it.

The voice side uses Whisper to transcribe what you said, then sends it to Llama 3.1 (running on NVIDIA NIM) which decides what tool to call. Tools include PowerShell commands, clicking things on screen, opening apps, typing text, reading files, and more. Responses come back in a British voice via edge-tts.

One thing I'm particularly happy with: Jarvis can look at your screen. If you say "click the subscribe button" it takes a screenshot, sends it to a vision model, figures out where the button is, and clicks the actual pixel coordinates. Works surprisingly well.

---

## Gestures

| Gesture | What it does |
|---|---|
| Open hand, move around | Move cursor |
| Thumb + index pinch | Left click |
| Thumb + middle pinch | Right click |
| Thumb + pinky pinch | Scroll — move hand up/down after pinching |
| Make a fist and hold | Start recording voice input |
| Release fist | Stop recording, process command |
| `T` key | Toggle the HUD overlay |
| `Esc` | Quit |

---

## What you can tell it to do

- Open apps (Chrome, VS Code, Spotify, etc.) or any website
- Run PowerShell commands
- Type text or press keyboard shortcuts
- Click, drag, or scroll anywhere
- Read what's on screen using computer vision, then interact with it
- Take screenshots, read and write files, manage the clipboard
- Search the web
- Save things it learns across sessions (stored in `lessons.json`)

It'll also tell you when it can't do something instead of lying about it, which took some prompt engineering to get right.

---

## Setup

**Requirements:** Windows 10/11, Python 3.10+, a webcam, an NVIDIA API key (free tier works fine — get one at [build.nvidia.com](https://build.nvidia.com))

### Quickest way

```bash
git clone https://github.com/T-Perm/Jarvis.py
cd Jarvis.py
setup.bat
```

`setup.bat` creates a venv, installs everything, downloads the hand landmark model, and sets up a `.env` template. Open `.env`, paste in your API key, then:

```bash
venv\Scripts\activate
python app.py
```

### Manual setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **Heads up:** this uses `pygame-ce` (Community Edition), not standard `pygame`. They conflict — only install one.

Download the hand model:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Copy `.env.example` to `.env` and fill in:

```env
NVIDIA_API_KEY=your_key_here
NIM_MODEL=meta/llama-3.1-8b-instruct
```

---

## Architecture

```
app.py          — main loop, gesture tracking, mouse control, HUD, fist trigger
Jarvis.py       — voice pipeline: STT → LLM → tool dispatch → TTS
hand_tracker.cc — C++ source for the GPU-accelerated DLL (optional)
```

For hand tracking there are two backends. If `hand_tracker.dll` is present it uses that — it's a compiled C++ binary with OpenCV and runs faster. If it's not there, it falls back to Python MediaPipe on CPU, which is fine for most uses.

The voice pipeline goes: sounddevice captures mic audio while you hold the fist → faster-whisper (base.en, CPU int8) transcribes it → NVIDIA NIM runs Llama 3.1 with tool calling → tools run locally (PowerShell, pyautogui, webbrowser, etc.) → edge-tts synthesizes the response and pygame plays it. The TTS is streamed sentence-by-sentence so the first sentence starts playing before the LLM finishes the full response.

The `read_screen` tool sends a screenshot to `nvidia/llama-3.2-11b-vision-instruct` and asks it to describe what's visible with pixel positions. Then Jarvis uses those coordinates to actually click.

You can swap the LLM by changing `NIM_MODEL` in your `.env`. Anything on NVIDIA NIM with tool-calling support should work.
