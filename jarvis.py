import os
import json
import threading
import subprocess
import webbrowser

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openai import OpenAI
import pyautogui

_nim = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")

_whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
SAMPLE_RATE = 16000

SYSTEM = (
    "You are J.A.R.V.I.S., a desktop AI assistant. "
    "Execute desktop commands using tools. Be concise."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open an application or file",
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
            "description": "Press a keyboard shortcut",
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
            "name": "click_screen",
            "description": "Click at absolute screen coordinates",
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
            "name": "close_window",
            "description": "Close the current window (Alt+F4)",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Print a spoken response to the user",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


class JarvisAgent:
    def __init__(self):
        self._frames = []
        self._stream = None
        self.status = "idle"

    @property
    def listening(self):
        return self._stream is not None

    def start_listening(self):
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._cb,
        )
        self._stream.start()
        self.status = "listening"

    def stop_and_process(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        threading.Thread(target=self._process, daemon=True).start()

    def _cb(self, indata, frames, time_, status):
        self._frames.append(indata.copy())

    def _process(self):
        self.status = "thinking"
        if not self._frames:
            self.status = "idle"
            return
        audio = np.concatenate(self._frames, axis=0).flatten()
        segments, _ = _whisper.transcribe(audio, beam_size=1)
        text = " ".join(s.text for s in segments).strip()
        if text:
            print(f"[STT] {text}")
            self._run_llm(text)
        self.status = "idle"

    def _dispatch(self, name, args):
        match name:
            case "open_app":
                try:
                    subprocess.Popen(["xdg-open", args["name"]])
                except Exception:
                    subprocess.Popen([args["name"]])
            case "type_text":
                pyautogui.typewrite(args["text"], interval=0.03)
            case "press_keys":
                pyautogui.hotkey(*args["keys"])
            case "search_web":
                webbrowser.open(f"https://google.com/search?q={args['query']}")
            case "click_screen":
                pyautogui.click(args["x"], args["y"])
            case "close_window":
                pyautogui.hotkey("alt", "F4")
            case "say":
                print(f"[JARVIS] {args['text']}")
        return "done"

    def _run_llm(self, text):
        messages = [
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
                break
            self.status = "acting"
            messages.append(msg)
            for tc in msg.tool_calls:
                result = self._dispatch(
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
