# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Jarvis — builds a single standalone Jarvis.exe.
#
#   venv\Scripts\pyinstaller Jarvis.spec --noconfirm
#
# A bundled Whisper model (whisper-base.en/) is included automatically if that
# folder exists next to this spec; otherwise the exe downloads base.en from
# Hugging Face on first run. See build_exe.bat to fetch the model beforehand.

import os
from PyInstaller.utils.hooks import collect_all

# --- Bundled native assets (loaded via ctypes / read by the C++ DLL) ---------
binaries = [
    ("hand_tracker.dll", "."),
    ("opencv_world3416.dll", "."),
]
datas = [
    ("hand_landmarker.task", "."),
]
hiddenimports = []

# Optionally bundle the Whisper model so the exe needs no first-run download.
# Only the files faster-whisper actually loads (skip the HF .cache/ and README).
if os.path.isdir("whisper-base.en"):
    for _f in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        _p = os.path.join("whisper-base.en", _f)
        if os.path.isfile(_p):
            datas.append((_p, "whisper-base.en"))

# --- Pull in data files / DLLs / submodules for the heavy native packages ----
for pkg in [
    "faster_whisper",
    "ctranslate2",
    "onnxruntime",
    "av",
    "tokenizers",
    "huggingface_hub",
    "sounddevice",
    "cv2",
    "pygame",
    "webrtcvad",
    "edge_tts",
    "openai",
    "certifi",
    "pyautogui",
    "mouse",
]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h


a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # The Python MediaPipe CPU fallback is dropped: the exe ships the C++ DLL
    # backend (hand_tracker.dll + opencv_world3416.dll) instead.
    excludes=["mediapipe"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Jarvis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
