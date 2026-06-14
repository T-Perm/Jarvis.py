import asyncio
import json
import os
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


def speak(text: str) -> None:
    def _worker() -> None:
        with _voice_lock:
            async def _synth() -> str:
                fd: int
                path: str
                fd, path = tempfile.mkstemp(suffix=".mp3")
                os.close(fd)
                await edge_tts.Communicate(text, JARVIS_VOICE).save(path)
                return path

            path: str = asyncio.run(_synth())
            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            finally:
                os.unlink(path)

    threading.Thread(target=_worker, daemon=True).start()


_nim: OpenAI = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
NIM_MODEL: str = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")

_whisper: WhisperModel = WhisperModel("base.en", device="cpu", compute_type="int8")
SAMPLE_RATE: int = 16000

SYSTEM: str = (
    "You are J.A.R.V.I.S., a desktop AI assistant with full control over the user's PC. "
    "Use tools to execute any request. Chain multiple tool calls as needed. "
    "run_command can do almost anything — use it freely. Be concise."
)

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
            "description": "Open an application by name or URL",
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
]


class JarvisAgent:
    _frames: list[np.ndarray]
    _stream: sd.InputStream | None
    status: str

    def __init__(self) -> None:
        self._frames = []
        self._stream = None
        self.status = "idle"

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
        self.status = "thinking"
        if not self._frames:
            self.status = "idle"
            return
        audio: np.ndarray = np.concatenate(self._frames, axis=0).flatten()
        segments, _ = _whisper.transcribe(audio, beam_size=1)
        text: str = " ".join(s.text for s in segments).strip()
        if text:
            print(f"[STT] {text}")
            self._run_llm(text)
        self.status = "idle"

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        match name:
            case "run_command":
                try:
                    result = subprocess.run(
                        ["powershell", "-NoProfile", "-Command", args["command"]],
                        capture_output=True, text=True, timeout=30,
                    )
                    output: str = (result.stdout + result.stderr).strip()
                    print(f"[CMD] {args['command'][:80]}\n{output[:300]}")
                    return output[:2000] or "done"
                except subprocess.TimeoutExpired:
                    return "timeout"
                except Exception as e:
                    return f"error: {e}"

            case "open_app":
                target: str = _APP_MAP.get(args["name"].lower().strip(), args["name"])
                if target.startswith("http"):
                    webbrowser.open(target)
                else:
                    exe: str = shutil.which(target) or target
                    try:
                        subprocess.Popen([exe])
                    except FileNotFoundError:
                        subprocess.Popen(exe, shell=True)

            case "type_text":
                pyautogui.typewrite(args["text"], interval=0.03)

            case "press_keys":
                pyautogui.hotkey(*args["keys"])

            case "click_screen":
                btn: str = args.get("button", "left")
                if args.get("double"):
                    pyautogui.doubleClick(args["x"], args["y"], button=btn)
                else:
                    pyautogui.click(args["x"], args["y"], button=btn)

            case "move_mouse":
                pyautogui.moveTo(args["x"], args["y"])

            case "drag":
                pyautogui.drag(
                    args["x2"] - args["x1"], args["y2"] - args["y1"],
                    duration=args.get("duration", 0.5),
                    button="left",
                )

            case "scroll":
                clicks: int = args.get("amount", 3)
                pyautogui.scroll(clicks if args["direction"] == "up" else -clicks)

            case "screenshot":
                path: str = args.get("path") or os.path.join(
                    os.path.expanduser("~"), "Desktop",
                    f"jarvis_{int(time.time())}.png",
                )
                pyautogui.screenshot(path)
                print(f"[screenshot] {path}")
                return path

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
                import tkinter as tk
                r: tk.Tk = tk.Tk()
                r.withdraw()
                r.clipboard_clear()
                r.clipboard_append(args["text"])
                r.update()
                r.after(500, r.destroy)
                r.mainloop()

            case "get_clipboard":
                import tkinter as tk
                r = tk.Tk()
                r.withdraw()
                try:
                    clipboard_text: str = r.clipboard_get()
                finally:
                    r.destroy()
                return clipboard_text

            case "search_web":
                webbrowser.open(f"https://google.com/search?q={args['query']}")

            case "say":
                print(f"[JARVIS] {args['text']}")
                speak(args["text"])

        return "done"

    def _run_llm(self, text: str) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ]
        while True:
            resp = _nim.chat.completions.create(
                model=NIM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                if msg.content:
                    print(f"[JARVIS] {msg.content}")
                    speak(msg.content)
                break
            self.status = "acting"
            messages.append(msg)
            for tc in msg.tool_calls:
                result: str = self._dispatch(
                    tc.function.name,
                    json.loads(tc.function.arguments),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result or "done",
                    }
                )
