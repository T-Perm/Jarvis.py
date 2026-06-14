# Jarvis — Hand-Tracked AI Desktop Controller

Control your PC with hand gestures and summon an AI agent with your voice. No mouse required.

## What it does

- **Tracks your hand** via webcam in real time (GPU-accelerated C++ DLL or CPU MediaPipe fallback)
- **Maps gestures to mouse actions** — move, left click, right click, scroll
- **Fist gesture** held for 0.5s activates voice input
- **Whisper** transcribes what you say
- **Llama 3.1** (via NVIDIA NIM) interprets the command and picks the right tool
- **Jarvis speaks back** in a British male voice (`en-GB-RyanNeural`)

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

- Run any PowerShell command
- Open apps (Chrome, VS Code, Spotify, Discord, etc.) or URLs
- Type text at the cursor
- Press keyboard shortcuts
- Click or drag anywhere on screen
- Take screenshots
- Read / write files
- Get or set clipboard contents
- Search the web
- Speak responses aloud

## Setup

### Requirements

- Windows 10/11
- Python 3.10+
- A webcam
- An NVIDIA API key (for the LLM — free tier available at build.nvidia.com)

### Install

```bash
python -m venv venv
venv\Scripts\activate
pip install opencv-python mediapipe faster-whisper openai sounddevice numpy mouse pyautogui python-dotenv edge-tts pygame
```

### Environment

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_key_here
NIM_MODEL=meta/llama-3.1-8b-instruct   # optional, this is the default
```

### Download the hand landmark model

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### Run

```bash
python app.py
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
4. Tools are dispatched locally (PowerShell, pyautogui, etc.)
5. `edge-tts` synthesizes the response as `en-GB-RyanNeural` MP3, played via `pygame`

## Notes

- The DLL backend requires `opencv_world3416.dll` alongside `hand_tracker.dll` in the project directory
- `hand_landmarker.task` and `*.dll` files are gitignored — download/build them separately
- The LLM model can be swapped via the `NIM_MODEL` env var to any model available on NVIDIA NIM
