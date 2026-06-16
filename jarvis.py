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
                pygame.mixer.music.unload()
                os.unlink(path)

    threading.Thread(target=_worker, daemon=True).start()


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
        try:
            if not self._frames:
                return
            audio: np.ndarray = np.concatenate(self._frames, axis=0).flatten()
            segments, _ = _whisper.transcribe(audio, beam_size=1)
            text: str = " ".join(s.text for s in segments).strip()
            if text:
                print(f"[STT] {text}")
                self._run_llm(text)
        except Exception as e:
            print(f"[JARVIS] error during processing: {e}")
        finally:
            self.status = "idle"

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
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
                    speak(args["text"])
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
                speak(reason)
                return "__cannot_do__"

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
            "3. You have NO vision by default. When the user asks you to click something, press a button, "
            "open a specific video, or interact with anything visible on screen, you MUST call read_screen "
            "first to see where things are.\n"
            "4. After read_screen, use click_screen with the actual pixel coordinates you observed.\n"
            "5. If you cannot complete a task (even after trying), you MUST call cannot_do — "
            "NEVER report false success.\n"
            "6. Use the learn tool to save what you discover across sessions.\n"
            "Be concise." + lesson_block
        )

    def _run_llm(self, text: str) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._build_system()},
            {"role": "user", "content": text},
        ]
        max_iterations: int = 10
        for _ in range(max_iterations):
            try:
                resp = _nim.chat.completions.create(
                    model=NIM_MODEL,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                )
            except Exception as e:
                print(f"[JARVIS] LLM request failed: {e}")
                speak("Sorry, I couldn't reach my language model.")
                return
            msg = resp.choices[0].message
            if not msg.tool_calls:
                if msg.content:
                    print(f"[JARVIS] {msg.content}")
                    speak(msg.content)
                return
            self.status = "acting"
            messages.append(msg)
            for tc in msg.tool_calls[:1]:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    print(f"[JARVIS] bad tool arguments for {tc.function.name}: {e}")
                    result: str = "error: invalid arguments"
                else:
                    result = self._dispatch(tc.function.name, args)
                if result == "__cannot_do__":
                    return
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result or "done",
                    }
                )
        print("[JARVIS] reached max tool iterations, stopping")
