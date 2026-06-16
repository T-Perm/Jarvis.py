import ctypes
import os
import time
import math
from dotenv import load_dotenv

load_dotenv(override=True)

import cv2
import numpy as np
import mouse
import pyautogui

from jarvis import JarvisAgent

MODEL_PATH: str = "hand_landmarker.task"
DLL_PATH: str = "hand_tracker.dll"
CAPTURE_W: int = 320
CAPTURE_H: int = 240
DISPLAY_W: int = 960
DISPLAY_H: int = 720
EMA_ALPHA: float = 0.35
PINCH_TRIGGER: float = 0.06
PINCH_RELEASE: float = 0.09
FIST_HOLD_S: float = 0.5
COOLDOWN_S: float = 0.2
FIST_PAIRS: list[tuple[int, int]] = [(8, 6), (12, 10), (16, 14), (20, 18)]
SCROLL_SCALE: int = 8

CYAN: tuple[int, int, int] = (255, 220, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
AMBER: tuple[int, int, int] = (0, 165, 255)
DIM_CYAN: tuple[int, int, int] = (180, 160, 0)


class _LM:
    __slots__ = ("x", "y", "z")
    x: float
    y: float
    z: float

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class DLLBackend:
    def __init__(self) -> None:
        _app_dir: str = os.path.dirname(os.path.abspath(__file__))
        os.add_dll_directory(_app_dir)
        ctypes.CDLL(os.path.join(_app_dir, "opencv_world3416.dll"))
        self._dll = ctypes.CDLL(os.path.join(_app_dir, DLL_PATH))
        self._dll.ht_start.restype = ctypes.c_int
        self._dll.ht_stop.restype = None
        self._dll.ht_get_landmarks.restype = ctypes.c_int
        self._dll.ht_get_landmarks.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int
        ]
        self._dll.ht_get_frame.restype = ctypes.c_int
        self._dll.ht_get_frame.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lm_buf: ctypes.Array[ctypes.c_double] = (ctypes.c_double * 63)()
        self._frame_buf: ctypes.Array[ctypes.c_uint8] = (ctypes.c_uint8 * (CAPTURE_W * CAPTURE_H * 3))()
        self._fw: ctypes.c_int = ctypes.c_int(0)
        self._fh: ctypes.c_int = ctypes.c_int(0)
        self._dll.ht_start()
        print("[tracker] C++ DLL backend active")

    def get_landmarks(self) -> list[_LM] | None:
        n: int = self._dll.ht_get_landmarks(self._lm_buf, 63)
        if n == 0:
            return None
        b = self._lm_buf
        return [_LM(float(b[i * 3]), float(b[i * 3 + 1]), float(b[i * 3 + 2])) for i in range(21)]

    def get_frame(self) -> np.ndarray | None:
        ok: int = self._dll.ht_get_frame(
            self._frame_buf,
            ctypes.byref(self._fw),
            ctypes.byref(self._fh),
        )
        if not ok:
            return None
        w: int = self._fw.value
        h: int = self._fh.value
        arr: np.ndarray = np.frombuffer(self._frame_buf, dtype=np.uint8)[: w * h * 3]
        return arr.reshape(h, w, 3).copy()

    def close(self) -> None:
        self._dll.ht_stop()


class PythonBackend:
    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._vis = mp_vision

        for delegate in (mp_tasks.BaseOptions.Delegate.GPU, mp_tasks.BaseOptions.Delegate.CPU):
            try:
                opts = mp_vision.HandLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(
                        model_asset_path=MODEL_PATH, delegate=delegate
                    ),
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                )
                self._lm = mp_vision.HandLandmarker.create_from_options(opts)
                break
            except Exception:
                continue

        self._cap: cv2.VideoCapture = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        self._frame: np.ndarray | None = None
        self._t0: float = time.time()
        print("[tracker] Python MediaPipe backend active (CPU)")

    def get_landmarks(self) -> list[_LM] | None:
        ok: bool
        frame: np.ndarray
        ok, frame = self._cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        self._frame = frame
        rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts: int = int((time.time() - self._t0) * 1000)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._lm.detect_for_video(mp_img, ts)
        if not result.hand_landmarks:
            return None
        lms = result.hand_landmarks[0]
        return [_LM(l.x, l.y, l.z) for l in lms]

    def get_frame(self) -> np.ndarray | None:
        return self._frame

    def close(self) -> None:
        self._cap.release()
        self._lm.close()


def build_backend() -> DLLBackend | PythonBackend:
    if os.path.exists(DLL_PATH):
        try:
            return DLLBackend()
        except Exception as e:
            print(f"[tracker] DLL failed ({e}), falling back to Python MediaPipe")
    return PythonBackend()


def dist3d(a: _LM, b: _LM) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def hysteresis(dist: float, trigger: float, release: float, active: bool) -> bool:
    if active:
        return dist < release
    return dist < trigger


def resolve_index_middle(lm: list[_LM]) -> tuple[_LM, _LM]:
    if dist3d(lm[8], lm[5]) < dist3d(lm[12], lm[5]):
        return lm[8], lm[12]
    return lm[12], lm[8]


def is_fist(lm: list[_LM]) -> bool:
    return all(lm[tip].y > lm[pip].y for tip, pip in FIST_PAIRS)


def draw_hud(
    frame: np.ndarray,
    status: str,
    sx: int,
    sy: int,
    gesture_label: str,
    frame_count: int,
) -> None:
    h: int
    w: int
    h, w = frame.shape[:2]
    overlay: np.ndarray = frame.copy()

    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    scan_y: int = (frame_count * 3) % h
    cv2.line(frame, (0, scan_y), (w, scan_y), DIM_CYAN, 1)

    arm: int = 20
    thickness: int = 2
    corners: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = [
        ((0, 0), (arm, 0), (0, arm)),
        ((w, 0), (w - arm, 0), (w, arm)),
        ((0, h), (arm, h), (0, h - arm)),
        ((w, h), (w - arm, h), (w, h - arm)),
    ]
    for corner, h_end, v_end in corners:
        cv2.line(frame, corner, h_end, CYAN, thickness)
        cv2.line(frame, corner, v_end, CYAN, thickness)

    title: str = "J.A.R.V.I.S."
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, title, ((w - tw) // 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    coords: str = f"X:{sx}  Y:{sy}"
    (cw, _), _ = cv2.getTextSize(coords, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(frame, coords, (w - cw - 8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

    if gesture_label:
        cv2.putText(frame, gesture_label, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA)

    if status == "listening":
        pulse_r: int = int(10 + 5 * math.sin(time.time() * 6))
        cv2.circle(frame, (w // 2, h // 2), pulse_r, CYAN, -1)
        cv2.putText(frame, "LISTENING...", (w // 2 - 55, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    elif status == "speaking":
        pulse_r = int(10 + 5 * math.sin(time.time() * 6))
        cv2.circle(frame, (w // 2, h // 2), pulse_r, CYAN, -1)
        cv2.putText(frame, "SPEAKING...", (w // 2 - 50, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    elif status in ("thinking", "acting", "transcribing", "calling jarvis") or status.startswith("running:"):
        amber_overlay: np.ndarray = frame.copy()
        cv2.rectangle(amber_overlay, (0, 0), (w, h), (0, 100, 200), -1)
        cv2.addWeighted(amber_overlay, 0.15, frame, 0.85, 0, frame)
        label: str = status.upper() + "..."
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, label, ((w - lw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA)


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Download it from:\n"
            "  https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    jarvis: JarvisAgent = JarvisAgent()
    tracker: DLLBackend | PythonBackend = build_backend()

    screen_w: int
    screen_h: int
    screen_w, screen_h = pyautogui.size()

    smooth_x: float = 0.0
    smooth_y: float = 0.0
    prev_lm9_y: float = 0.0
    left_down: bool = False
    right_down: bool = False
    scroll_active: bool = False
    last_fire: dict[str, float] = {"left": 0.0, "right": 0.0}
    fist_start: float | None = None
    show_window: bool = True
    frame_count: int = 0

    while True:
        lm_pts: list[_LM] | None = tracker.get_landmarks()
        frame: np.ndarray | None = tracker.get_frame()

        if frame is None and lm_pts is None and isinstance(tracker, PythonBackend):
            break

        if frame is None:
            frame = np.zeros((CAPTURE_H, CAPTURE_W, 3), dtype=np.uint8)

        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        gesture_label: str = ""

        if lm_pts is not None:
            sx: float = lm_pts[9].x * screen_w
            sy: float = lm_pts[9].y * screen_h
            smooth_x = EMA_ALPHA * sx + (1 - EMA_ALPHA) * smooth_x
            smooth_y = EMA_ALPHA * sy + (1 - EMA_ALPHA) * smooth_y
            mouse.move(int(smooth_x), int(smooth_y), absolute=True)

            index_tip: _LM
            middle_tip: _LM
            index_tip, middle_tip = resolve_index_middle(lm_pts)

            d_pi: float = dist3d(lm_pts[4], lm_pts[20])
            scroll_active = hysteresis(d_pi, PINCH_TRIGGER, PINCH_RELEASE, scroll_active)

            if scroll_active:
                delta: float = (prev_lm9_y - smooth_y) * SCROLL_SCALE
                if int(delta):
                    mouse.wheel(int(delta))
                gesture_label = "SCROLL"
            else:
                d_li: float = dist3d(lm_pts[4], index_tip)
                new_left: bool = hysteresis(d_li, PINCH_TRIGGER, PINCH_RELEASE, left_down)
                now: float = time.time()
                if new_left and not left_down and (now - last_fire["left"]) >= COOLDOWN_S:
                    mouse.press("left")
                    left_down = True
                    last_fire["left"] = now
                    gesture_label = "L-CLICK"
                elif not new_left and left_down:
                    mouse.release("left")
                    left_down = False
                elif left_down:
                    gesture_label = "L-CLICK"

                d_rm: float = dist3d(lm_pts[4], middle_tip)
                new_right: bool = hysteresis(d_rm, PINCH_TRIGGER, PINCH_RELEASE, right_down)
                if new_right and not right_down and (now - last_fire["right"]) >= COOLDOWN_S:
                    mouse.press("right")
                    right_down = True
                    last_fire["right"] = now
                    gesture_label = "R-CLICK"
                elif not new_right and right_down:
                    mouse.release("right")
                    right_down = False
                elif right_down:
                    gesture_label = "R-CLICK"

            prev_lm9_y = smooth_y

            now = time.time()
            if is_fist(lm_pts):
                if fist_start is None:
                    fist_start = now
                elif fist_start != -1 and (now - fist_start) >= FIST_HOLD_S and not jarvis.listening:
                    jarvis.start_listening()
                    fist_start = -1
            else:
                if jarvis.listening:
                    jarvis.stop_and_process()
                fist_start = None
        else:
            if left_down:
                mouse.release("left")
                left_down = False
            if right_down:
                mouse.release("right")
                right_down = False

        draw_hud(frame, jarvis.status, int(smooth_x), int(smooth_y), gesture_label, frame_count)

        if show_window:
            cv2.imshow("J.A.R.V.I.S.", frame)

        key: int = cv2.waitKey(1) & 0xFF
        if key == ord("t"):
            show_window = not show_window
            if not show_window:
                cv2.destroyAllWindows()
        elif key == 27:
            break

        frame_count += 1

    if left_down:
        mouse.release("left")
    if right_down:
        mouse.release("right")
    if jarvis.listening:
        jarvis.stop_and_process()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
