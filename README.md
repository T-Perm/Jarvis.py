# Jarvis — Hand-Tracked AI Desktop Controller

Control your PC with hand gestures and talk to an AI agent. No mouse required (mostly).

I made this because I got curious one weekend whether I could ditch my mouse and just use my webcam instead. Short answer: yeah, kind of! It's not going to replace a real mouse for precision stuff, but for moving around, clicking, and barking commands at it like Tony Stark, it actually works.

---

## How it works

You hold your hand up in front of the webcam and Jarvis tracks it. Move your hand, the cursor moves. Pinch your thumb and index finger together to click. Make a fist and hold it for half a second and it starts listening — say what you want, open your hand again, and it goes off and does it.

The voice part uses Whisper to figure out what you said, then hands it to Llama 3.1 (running on NVIDIA's NIM cloud thing) which decides which tool to call. The tools let it run PowerShell, click stuff, open apps, type text, read files, and a bunch more. It talks back in a British voice through edge-tts, because obviously Jarvis has to be British.

The part I'm most proud of: it can actually *look* at your screen. If you say "click the subscribe button," it grabs a screenshot, sends it to a vision model, gets back roughly where the button is, and clicks those pixels. I genuinely did not expect this to work and then it did.

---

## Gestures

| Gesture | What it does |
|---|---|
| Open hand, move around | Move cursor |
| Thumb + index pinch | Left click |
| Thumb + middle pinch | Right click |
| Thumb + pinky pinch | Scroll — move hand up/down after pinching |
| Make a fist and hold | Start recording voice input |
| Release fist | Stop recording, run the command |
| `T` key | Toggle the camera/HUD overlay |
| `Esc` | Quit |

---

## Stuff you can tell it to do

- Open apps (Chrome, VS Code, Spotify, whatever) or any website
- Run PowerShell commands
- Type text or hit keyboard shortcuts
- Click, drag, or scroll anywhere
- Look at what's on screen and then actually interact with it
- Take screenshots, read/write files, mess with the clipboard
- Search the web
- Remember things between sessions (it dumps them in `lessons.json`)

It also tells you when it *can't* do something instead of pretending it worked, which sounds simple but took me way too many tries with the prompt to get right. Early versions would just confidently lie.

---

## Setup

**You'll need:** Windows 10/11, a webcam, and a free NVIDIA API key (grab one at [build.nvidia.com](https://build.nvidia.com) — the free tier is plenty). If you're running from source you also need Python 3.10+.

### Easiest option — just run the .exe

`Jarvis.exe` is one single file with everything packed inside it (the hand-tracking DLL, OpenCV, the hand-landmark model, and the Whisper speech model). No Python, no pip, no setup. Build it yourself with [`build_exe.bat`](#building-the-exe), or download it from the releases if I've put one up.

1. Drop `Jarvis.exe` in its own folder.
2. Make a `.env` file in that same folder:

   ```env
   NVIDIA_API_KEY=your_key_here
   NIM_MODEL=meta/llama-3.1-8b-instruct
   ```
3. Double-click it. Heads up — the first launch takes like 15-20 seconds because it has to unpack all the bundled stuff to a temp folder. It's not frozen, just be patient.

It writes `lessons.json` next to the exe so it remembers things between runs.

### Running from source

```bash
git clone https://github.com/T-Perm/Jarvis.py
cd Jarvis.py
setup.bat
```

`setup.bat` makes a venv, installs everything, grabs the hand landmark model, and drops a `.env` template. Open `.env`, paste your key in, then:

```bash
venv\Scripts\activate
python app.py
```

If you'd rather do it by hand:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> **One gotcha that cost me an afternoon:** this uses `pygame-ce` (Community Edition), NOT regular `pygame`. They clash with each other, so only install one of them.

Then grab the hand model:

```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

And copy `.env.example` to `.env` and fill it in:

```env
NVIDIA_API_KEY=your_key_here
NIM_MODEL=meta/llama-3.1-8b-instruct
```

---

## How it's put together

```
app.py          — main loop, gesture tracking, mouse control, the HUD, fist trigger
jarvis.py       — the voice side: speech-to-text → LLM → tool calls → text-to-speech
hand_tracker.cc — C++ source for the faster DLL backend (optional)
```

There are two ways it tracks your hand. If `hand_tracker.dll` is sitting there it uses that — it's a compiled C++ binary using OpenCV and it's noticeably faster. If the DLL's missing it falls back to plain Python MediaPipe on the CPU, which is slower but works fine.

The voice pipeline roughly goes: sounddevice records the mic while you're holding a fist → faster-whisper (base.en, int8 on CPU) turns it into text → NVIDIA NIM runs Llama 3.1 with tool calling → the tools run locally (PowerShell, pyautogui, webbrowser, etc.) → edge-tts makes the audio and pygame plays it. The speech gets streamed sentence by sentence, so it starts talking before the model has even finished writing the whole reply. That one took some fiddling but it makes it feel way snappier.

The `read_screen` tool sends a screenshot to `nvidia/llama-3.2-11b-vision-instruct` and asks it to describe what's on screen with pixel positions, and then Jarvis clicks based on those coordinates.

Want a different model? Just change `NIM_MODEL` in your `.env`. Anything on NVIDIA NIM that supports tool calling should work.

---

## Building the exe

The standalone `Jarvis.exe` is built with [PyInstaller](https://pyinstaller.org) using `Jarvis.spec`. Once you've run `setup.bat`:

```bash
build_exe.bat
```

That downloads the Whisper `base.en` model, bundles it together with `hand_tracker.dll`, `opencv_world3416.dll`, and `hand_landmarker.task`, and spits out `dist\Jarvis.exe` — one file, around 336 MB. It's chunky because the speech model alone is ~140 MB, but the upside is it just runs anywhere with no install.

A couple of things worth knowing: the exe only ships the fast C++ DLL backend (I left the Python MediaPipe fallback out to keep the size down), so the bundled DLL — and the Visual C++ runtime it depends on — needs to work on whatever machine you run it on. And if you want a smaller exe that just downloads the speech model the first time it runs instead of carrying it around, delete the `whisper-base.en` folder before building.
