# Dev Journal — 2026-06-14

## Session: Voice Replacement & First Commit

### What we were working on
A hand-tracking desktop controller (`app.py`) that uses MediaPipe/a custom C++ DLL to track hand gestures via webcam and map them to mouse actions (move, left click, right click, scroll). A fist gesture held for 0.5s summons a Jarvis-style AI agent (`jarvis.py`) that listens for a voice command, transcribes it with Whisper, and sends it to an LLM (NVIDIA NIM / Llama 3.1) which can execute PowerShell commands, open apps, type text, click the screen, manage files, and speak responses aloud.

---

### What we did

#### 1. Diagnosed the voice
The original TTS setup in `jarvis.py` used **NVIDIA Riva** (cloud gRPC API) with the voice `Magpie-Multilingual.EN-US.Leo` — a generic American male voice. It required an `NVIDIA_API_KEY` and streamed audio over gRPC.

The goal: replace it with something that actually sounds like Jarvis.

#### 2. Picked a replacement voice
After looking at the options, landed on **Microsoft Edge TTS** (`edge-tts` Python package) with the voice `en-GB-RyanNeural`:
- British RP (Received Pronunciation) accent — the defining characteristic of the Jarvis character
- Deep, authoritative, clear
- Free, no API key needed — uses the same engine as the Edge browser's read-aloud feature
- Widely regarded in the maker/AI community as the closest free approximation to the Jarvis voice

Alternatives considered and rejected:
- OpenAI TTS `onyx` — good but costs money per character
- Windows SAPI / pyttsx3 — available built-in but lower quality
- Staying on Riva with a British voice — unclear which exact EN-GB voice names were available

#### 3. Rewrote the `speak()` function (`jarvis.py`)
Replaced the entire Riva TTS block with an `edge-tts` + `pygame` implementation:

**Before:**
- NVIDIA Riva gRPC client, streaming PCM chunks into `sounddevice.OutputStream`
- Required `NVIDIA_API_KEY`, `riva.client` package

**After:**
- `edge_tts.Communicate(text, "en-GB-RyanNeural").save(path)` — saves MP3 to a temp file
- `pygame.mixer.music` plays it back, blocks until done
- Cleaned up stale imports (`queue`, `riva.client`)

New dependencies: `edge-tts`, `pygame`

#### 4. Committed and pushed
Staged all project files — `app.py`, `jarvis.py`, `.gitignore`, `hand_tracker.cc` — and pushed to:

> https://github.com/T-Perm/Jarvis.py  
> Commit: `aab99f3` — *"Created agent that orchastrates commands via voice input from the user"*

This was the first real commit on top of the initial skeleton.

---

### State of the project at commit
| File | Status |
|---|---|
| `app.py` | Main loop — gesture tracking, HUD rendering, fist-to-summon trigger |
| `jarvis.py` | Agent — STT (Whisper), LLM (Llama 3.1 via NVIDIA NIM), TTS (Edge RyanNeural), tool dispatch |
| `hand_tracker.cc` | C++ source for the GPU-accelerated DLL backend |
| `.gitignore` | Standard ignores |

### What's next (potential)
- Test the new voice end-to-end
- Tune fist hold threshold (`FIST_HOLD_S`) for comfort
- Explore upgrading the LLM model (`NIM_MODEL` env var)
- Add a requirements.txt
