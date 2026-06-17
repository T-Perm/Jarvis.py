import asyncio
import json
import os
import queue as _q
import re
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser
from typing import Any

import numpy as np
import sounddevice as sd
import edge_tts
import pygame
import pyautogui
from faster_whisper import WhisperModel
from openai import OpenAI

JARVIS_VOICE: str = "en-GB-RyanNeural"
_voice_lock: threading.Lock = threading.Lock()

pygame.mixer.init()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _launch_sentence_player(sentences_q: _q.Queue, on_speaking=None, on_idle=None) -> None:
    """Synthesize sentences from sentences_q one by one, play them in order.

    Put sentence strings into sentences_q; put None to signal end.
    Synthesis of sentence N+1 overlaps with playback of sentence N.
    """
    def _worker() -> None:
        if on_speaking:
            on_speaking()

        audio_q: _q.Queue = _q.Queue()

        async def _synth_loop() -> None:
            loop = asyncio.get_running_loop()
            while True:
                sentence: str | None = await loop.run_in_executor(None, sentences_q.get)
                if sentence is None:
                    audio_q.put(None)
                    return
                fd, path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                try:
                    await edge_tts.Communicate(sentence, JARVIS_VOICE).save(path)
                    audio_q.put(path)
                except Exception as e:
                    print(f"[TTS] synth error: {e}")
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        synth_t = threading.Thread(target=lambda: asyncio.run(_synth_loop()), daemon=True)
        synth_t.start()

        played: list[str] = []
        with _voice_lock:
            try:
                while True:
                    path: str | None = audio_q.get()
                    if path is None:
                        break
                    played.append(path)
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    pygame.mixer.music.unload()
            finally:
                for p in played:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

        synth_t.join(timeout=10)
        if on_idle:
            on_idle()

    threading.Thread(target=_worker, daemon=True).start()


def speak_pipeline(text: str, on_speaking=None, on_idle=None) -> None:
    sentences = _split_sentences(text)
    if not sentences:
        return
    q: _q.Queue = _q.Queue()
    for s in sentences:
        q.put(s)
    q.put(None)
    _launch_sentence_player(q, on_speaking=on_speaking, on_idle=on_idle)


_nim: OpenAI = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
NIM_MODEL: str = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")

_whisper: WhisperModel = WhisperModel("base.en", device="cpu", compute_type="int8")
SAMPLE_RATE: int = 16000

LESSONS_FILE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lessons.json")


def _load_lessons() -> list[dict[str, str]]:
    try:
        with open(LESSONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_lesson(intent: str, lesson: str) -> None:
    lessons = _load_lessons()
    lessons.append({"intent": intent, "lesson": lesson})
    with open(LESSONS_FILE, "w", encoding="utf-8") as f:
        json.dump(lessons, f, indent=2)

_APP_MAP: dict[str, str] = {
    "chrome": "chrome", "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge", "microsoft edge": "msedge",
    "brave": "brave",
    "notepad": "notepad",
    "calculator": "calc",
    "paint": "mspaint",
    "explorer": "explorer", "file explorer": "explorer",
    "task manager": "taskmgr",
    "cmd": "cmd", "command prompt": "cmd",
    "powershell": "powershell",
    "word": "winword", "excel": "excel", "powerpoint": "powerpnt",
    "outlook": "outlook",
    "vscode": "code", "vs code": "code", "visual studio code": "code",
    "discord": "discord",
    "spotify": "spotify",
    "steam": "steam",
    "teams": "teams",
    "gmail": "https://mail.google.com",
    "email": "https://mail.google.com",
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "reddit": "https://www.reddit.com",
    "netflix": "https://www.netflix.com",
    "chatgpt": "https://chat.openai.com",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a PowerShell command and return its output. Use for anything: file ops, installs, system info, registry, networking, etc.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "PowerShell command to execute"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application by name, or open any website URL or domain (e.g. 'hackclub.com', 'https://github.com'). Always use this for websites — never run_command.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text at the current cursor position",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_keys",
            "description": "Press a keyboard shortcut or key combination (e.g. ['ctrl','c'], ['alt','F4'], ['win','d'])",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_screen",
            "description": "Click at absolute screen coordinates",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "enum": ["left", "right", "middle"], "default": "left"},
                    "double": {"type": "boolean", "default": False},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_mouse",
            "description": "Move mouse to absolute screen coordinates without clicking",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drag",
            "description": "Click and drag from one screen position to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "x1": {"type": "integer"}, "y1": {"type": "integer"},
                    "x2": {"type": "integer"}, "y2": {"type": "integer"},
                    "duration": {"type": "number", "default": 0.5},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scroll",
            "description": "Scroll the mouse wheel at current position",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "amount": {"type": "integer", "default": 3},
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "Take a screenshot and save it. Returns the file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Save path (optional, defaults to Desktop)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (overwrites if exists)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_clipboard",
            "description": "Copy text to the clipboard",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_clipboard",
            "description": "Read the current clipboard contents",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Open a web browser search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Speak a response aloud to the user",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "learn",
            "description": (
                "Save a lesson for future use. Call this when you discover what approach "
                "works or fails for a given type of request — good or bad outcomes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "description": "Short label for the request type, e.g. 'open website', 'launch app by name'"},
                    "lesson": {"type": "string", "description": "What you learned: what worked, what failed, the correct approach to use next time"},
                },
                "required": ["intent", "lesson"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_screen",
            "description": (
                "Take a screenshot and use computer vision to describe what is currently on screen. "
                "Call this BEFORE any click, button press, or visual interaction so you know where things are. "
                "Returns positions and descriptions of visible elements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What to look for on screen, e.g. 'Where is the first video thumbnail and what are its coordinates?'",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cannot_do",
            "description": (
                "Call this when you have determined you cannot complete the user's request. "
                "This speaks the reason aloud. You MUST call this instead of pretending to succeed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Clear explanation of why the task cannot be completed and what would be needed.",
                    }
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_conversation",
            "description": "Clear conversation memory. Call this when the user asks you to start fresh, forget everything, or reset the conversation.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class JarvisAgent:
    _frames: list[np.ndarray]
    _stream: sd.InputStream | None
    status: str

    def __init__(self) -> None:
        self._frames = []
        self._stream = None
        self.status = "idle"
        self._history: list[dict] = []
        self._last_command_time: float = 0.0

    def _speak(self, text: str) -> None:
        speak_pipeline(text,
                       on_speaking=lambda: setattr(self, "status", "speaking"),
                       on_idle=lambda: setattr(self, "status", "idle"))

    @property
    def listening(self) -> bool:
        return self._stream is not None

    def start_listening(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._cb,
        )
        self._stream.start()
        self.status = "listening"

    def stop_and_process(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        threading.Thread(target=self._process, daemon=True).start()

    def _cb(self, indata: np.ndarray, frames: int, time_: Any, status: sd.CallbackFlags) -> None:
        self._frames.append(indata.copy())

    def _process(self) -> None:
        self.status = "transcribing"
        try:
            if not self._frames:
                self.status = "idle"
                return
            audio: np.ndarray = np.concatenate(self._frames, axis=0).flatten()
            segments, _ = _whisper.transcribe(audio, beam_size=1)
            text: str = " ".join(s.text for s in segments).strip()
            if not text:
                self.status = "idle"
                return
            print(f"[STT] {text}")
            self._run_llm(text)
        except Exception as e:
            print(f"[JARVIS] error during processing: {e}")
            self.status = "idle"

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        self.status = f"running: {name}"
        match name:
            case "run_command":
                cmd: str = args["command"].strip()
                cmd = cmd.replace('\\"', '"')
                cmd_lower = cmd.lower()
                if cmd_lower.startswith("http://") or cmd_lower.startswith("https://") or (
                    "." in cmd and " " not in cmd and not cmd.startswith("-") and
                    cmd.split(".")[-1] in {"com", "org", "net", "io", "dev", "co", "ai", "app", "gov", "edu"}
                ):
                    url = cmd if cmd_lower.startswith("http") else "https://" + cmd
                    webbrowser.open(url)
                    return "[ok]\nopened in browser"
                try:
                    result = subprocess.run(
                        ["powershell", "-NoProfile", "-Command", cmd],
                        capture_output=True, text=True, timeout=30,
                    )
                    output: str = (result.stdout + result.stderr).strip()
                    status: str = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
                    print(f"[CMD {status}] {cmd[:80]}\n{output[:300]}")
                    return (f"[{status}]\n{output}")[:2000] or f"[{status}]"
                except subprocess.TimeoutExpired:
                    return "timeout"
                except Exception as e:
                    return f"error: {e}"

            case "open_app":
                name_clean: str = args["name"].lower().strip()
                target: str = _APP_MAP.get(name_clean, args["name"])
                if not target.startswith("http") and "." in target and " " not in target:
                    target = "https://" + target
                try:
                    if target.startswith("http"):
                        webbrowser.open(target)
                    else:
                        exe: str = shutil.which(target) or target
                        try:
                            subprocess.Popen([exe])
                        except FileNotFoundError:
                            subprocess.Popen(exe, shell=True)
                except Exception as e:
                    return f"error: {e}"

            case "type_text":
                try:
                    pyautogui.typewrite(args["text"], interval=0.03)
                except Exception as e:
                    return f"error: {e}"

            case "press_keys":
                try:
                    pyautogui.hotkey(*args["keys"])
                except Exception as e:
                    return f"error: {e}"

            case "click_screen":
                try:
                    btn: str = args.get("button", "left")
                    if args.get("double"):
                        pyautogui.doubleClick(args["x"], args["y"], button=btn)
                    else:
                        pyautogui.click(args["x"], args["y"], button=btn)
                except Exception as e:
                    return f"error: {e}"

            case "move_mouse":
                try:
                    pyautogui.moveTo(args["x"], args["y"])
                except Exception as e:
                    return f"error: {e}"

            case "drag":
                try:
                    pyautogui.drag(
                        args["x2"] - args["x1"], args["y2"] - args["y1"],
                        duration=args.get("duration", 0.5),
                        button="left",
                    )
                except Exception as e:
                    return f"error: {e}"

            case "scroll":
                try:
                    clicks: int = args.get("amount", 3)
                    pyautogui.scroll(clicks if args["direction"] == "up" else -clicks)
                except Exception as e:
                    return f"error: {e}"

            case "screenshot":
                try:
                    path: str = args.get("path") or os.path.join(
                        os.path.expanduser("~"), "Desktop",
                        f"jarvis_{int(time.time())}.png",
                    )
                    pyautogui.screenshot(path)
                    print(f"[screenshot] {path}")
                    return path
                except Exception as e:
                    return f"error: {e}"

            case "read_file":
                try:
                    with open(args["path"], "r", encoding="utf-8", errors="replace") as f:
                        content: str = f.read(8000)
                    return content
                except Exception as e:
                    return f"error: {e}"

            case "write_file":
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(args["path"])), exist_ok=True)
                    with open(args["path"], "w", encoding="utf-8") as f:
                        f.write(args["content"])
                    return "written"
                except Exception as e:
                    return f"error: {e}"

            case "set_clipboard":
                try:
                    import tkinter as tk
                    r: tk.Tk = tk.Tk()
                    r.withdraw()
                    r.clipboard_clear()
                    r.clipboard_append(args["text"])
                    r.update()
                    r.after(500, r.destroy)
                    r.mainloop()
                except Exception as e:
                    return f"error: {e}"

            case "get_clipboard":
                try:
                    import tkinter as tk
                    r = tk.Tk()
                    r.withdraw()
                    try:
                        clipboard_text: str = r.clipboard_get()
                    finally:
                        r.destroy()
                    return clipboard_text
                except Exception as e:
                    return f"error: {e}"

            case "search_web":
                try:
                    webbrowser.open(f"https://google.com/search?q={args['query']}")
                except Exception as e:
                    return f"error: {e}"

            case "say":
                try:
                    print(f"[JARVIS] {args['text']}")
                    self._speak(args["text"])
                except Exception as e:
                    return f"error: {e}"

            case "learn":
                try:
                    _save_lesson(args["intent"], args["lesson"])
                    print(f"[JARVIS] learned: [{args['intent']}] {args['lesson']}")
                    return "lesson saved"
                except Exception as e:
                    return f"error: {e}"

            case "read_screen":
                import base64
                import io
                question: str = args.get(
                    "question",
                    "Describe everything visible: windows, buttons, text, videos, and their approximate pixel positions.",
                )
                try:
                    img = pyautogui.screenshot()
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    b64: str = base64.b64encode(buf.getvalue()).decode()
                    vision_resp = _nim.chat.completions.create(
                        model="nvidia/llama-3.2-11b-vision-instruct",
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                                {"type": "text", "text": question},
                            ],
                        }],
                        max_tokens=600,
                    )
                    description: str = vision_resp.choices[0].message.content
                    print(f"[SCREEN] {description[:300]}")
                    return description
                except Exception as e:
                    return f"error reading screen: {e}"

            case "cannot_do":
                reason: str = args.get("reason", "I don't know how to do that.")
                print(f"[JARVIS] cannot_do: {reason}")
                self._speak(reason)
                return "__cannot_do__"

            case "reset_conversation":
                self._history.clear()
                self._speak("Memory cleared.")
                return "done"

            case _:
                print(f"[JARVIS] unknown tool requested: {name}")
                return f"error: unknown tool '{name}'"

        return "done"

    def _build_system(self) -> str:
        lessons = _load_lessons()
        lesson_block = ""
        if lessons:
            lines = "\n".join(f"- [{l['intent']}] {l['lesson']}" for l in lessons[-30:])
            lesson_block = f"\n\nPast lessons (use these to inform your approach):\n{lines}"
        return (
            "You are J.A.R.V.I.S., a desktop AI assistant. "
            "RULES YOU MUST FOLLOW:\n"
            "1. For websites/URLs/domains, use open_app — NEVER run_command.\n"
            "2. run_command returns [ok] or [FAILED (exit N)] — check this before claiming success.\n"
            "3. For live internet data (stock prices, weather, news, sports scores, exchange rates), "
            "use search_web — PowerShell has no internet access and no live data cmdlets.\n"
            "4. You have NO vision by default. When the user asks you to click something, press a button, "
            "open a specific video, or interact with anything visible on screen, you MUST call read_screen "
            "first to see where things are.\n"
            "5. After read_screen, use click_screen with the actual pixel coordinates you observed.\n"
            "6. If you cannot complete a task (even after trying), you MUST call cannot_do — "
            "NEVER report false success.\n"
            "7. Use the learn tool to save what you discover across sessions.\n"
            "Be concise." + lesson_block
        )

    def _run_llm(self, text: str) -> None:
        now = time.time()
        if self._last_command_time and (now - self._last_command_time) > 300:
            self._history.clear()
        self._last_command_time = now

        history_len = len(self._history)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system()},
            *self._history,
            {"role": "user", "content": text},
        ]
        max_iterations: int = 10
        try:
            for _ in range(max_iterations):
                self.status = "calling jarvis"
                try:
                    stream = _nim.chat.completions.create(
                        model=NIM_MODEL,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        parallel_tool_calls=False,
                        stream=True,
                    )
                except Exception as e:
                    print(f"[JARVIS] LLM request failed: {e}")
                    self._speak("Sorry, I couldn't reach my language model.")
                    return

                full_content: str = ""
                tool_calls_acc: dict[int, dict[str, str]] = {}
                sentence_buf: str = ""
                live_tts_q: _q.Queue | None = None

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    if delta.content:
                        full_content += delta.content
                        sentence_buf += delta.content
                        while True:
                            m = re.search(r'(?<=[.!?])\s', sentence_buf)
                            if not m:
                                break
                            sentence = sentence_buf[: m.start() + 1].strip()
                            sentence_buf = sentence_buf[m.end():]
                            if not sentence:
                                continue
                            if live_tts_q is None:
                                live_tts_q = _q.Queue()
                                _launch_sentence_player(
                                    live_tts_q,
                                    on_speaking=lambda: setattr(self, "status", "speaking"),
                                    on_idle=lambda: setattr(self, "status", "idle"),
                                )
                            live_tts_q.put(sentence)

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {"id": "", "name": "", "args": ""}
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                tool_calls_acc[idx]["name"] += tc_delta.function.name or ""
                                tool_calls_acc[idx]["args"] += tc_delta.function.arguments or ""

                # Flush any remaining text not yet delimited
                if sentence_buf.strip():
                    if live_tts_q is None:
                        live_tts_q = _q.Queue()
                        _launch_sentence_player(
                            live_tts_q,
                            on_speaking=lambda: setattr(self, "status", "speaking"),
                            on_idle=lambda: setattr(self, "status", "idle"),
                        )
                    live_tts_q.put(sentence_buf.strip())
                if live_tts_q is not None:
                    live_tts_q.put(None)

                # Reconstruct message dict for history
                if tool_calls_acc:
                    tc_list = [
                        {
                            "id": tool_calls_acc[i]["id"],
                            "type": "function",
                            "function": {
                                "name": tool_calls_acc[i]["name"],
                                "arguments": tool_calls_acc[i]["args"],
                            },
                        }
                        for i in sorted(tool_calls_acc)
                    ]
                    msg_dict: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": tc_list}
                else:
                    msg_dict = {"role": "assistant", "content": full_content}
                messages.append(msg_dict)

                if not tool_calls_acc:
                    print(f"[JARVIS] {full_content}")
                    if full_content and live_tts_q is None:
                        self._speak(full_content)
                    elif not full_content:
                        self.status = "idle"
                    return

                # Dispatch the first (and only) tool call
                tc_data = tool_calls_acc[min(tool_calls_acc)]
                try:
                    args = json.loads(tc_data["args"])
                except json.JSONDecodeError as e:
                    print(f"[JARVIS] bad tool arguments for {tc_data['name']}: {e}")
                    result: str = "error: invalid arguments"
                else:
                    result = self._dispatch(tc_data["name"], args)
                if result == "__cannot_do__":
                    return
                messages.append({"role": "tool", "tool_call_id": tc_data["id"], "content": result or "done"})

            self.status = "idle"
            print("[JARVIS] reached max tool iterations, stopping")
        finally:
            new_turns = messages[history_len + 1:]
            self._history.extend(new_turns)
            if len(self._history) > 20:
                self._history = self._history[-20:]
