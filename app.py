import os
import time
import math

import cv2
import numpy as np
import mouse
import pyautogui
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from jarvis import JarvisAgent

MODEL_PATH = "hand_landmarker.task"
CAPTURE_W, CAPTURE_H = 320, 240
EMA_ALPHA = 0.35
PINCH_TRIGGER = 0.06
PINCH_RELEASE = 0.09
FIST_HOLD_S = 0.5
COOLDOWN_S = 0.2
FIST_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]
SCROLL_SCALE = 8

# Jarvis HUD colours (BGR)
CYAN = (255, 220, 0)
WHITE = (255, 255, 255)
AMBER = (0, 165, 255)
DIM_CYAN = (180, 160, 0)


def build_landmarker():
    for delegate in (
        mp_tasks.BaseOptions.Delegate.GPU,
        mp_tasks.BaseOptions.Delegate.CPU,
    ):
        try:
            opts = mp_vision.HandLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=MODEL_PATH, delegate=delegate
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
            )
            return mp_vision.HandLandmarker.create_from_options(opts)
        except Exception:
            continue
    raise RuntimeError("Cannot create HandLandmarker on GPU or CPU")


def dist3d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def hysteresis(dist, trigger, release, active):
    if active:
        return dist < release
    return dist < trigger


def resolve_index_middle(lm):
    if dist3d(lm[8], lm[5]) < dist3d(lm[12], lm[5]):
        return lm[8], lm[12]
    return lm[12], lm[8]


def is_fist(lm):
    return all(lm[tip].y > lm[pip].y for tip, pip in FIST_PAIRS)


def draw_hud(frame, status, sx, sy, gesture_label, frame_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark background blend
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # Animated scan line
    scan_y = (frame_count * 3) % h
    cv2.line(frame, (0, scan_y), (w, scan_y), DIM_CYAN, 1)

    # Cyan corner brackets
    arm = 20
    thickness = 2
    corners = [
        ((0, 0), (arm, 0), (0, arm)),
        ((w, 0), (w - arm, 0), (w, arm)),
        ((0, h), (arm, h), (0, h - arm)),
        ((w, h), (w - arm, h), (w, h - arm)),
    ]
    for corner, h_end, v_end in corners:
        cv2.line(frame, corner, h_end, CYAN, thickness)
        cv2.line(frame, corner, v_end, CYAN, thickness)

    # Title
    title = "J.A.R.V.I.S."
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        frame, title, ((w - tw) // 2, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA,
    )

    # Cursor coords bottom-right
    coords = f"X:{sx}  Y:{sy}"
    (cw, _), _ = cv2.getTextSize(coords, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    cv2.putText(
        frame, coords, (w - cw - 8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA,
    )

    # Gesture label bottom-left
    if gesture_label:
        cv2.putText(
            frame, gesture_label, (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN, 1, cv2.LINE_AA,
        )

    # Status overlays
    if status == "listening":
        pulse_r = int(10 + 5 * math.sin(time.time() * 6))
        cv2.circle(frame, (w // 2, h // 2), pulse_r, CYAN, -1)
        cv2.putText(
            frame, "LISTENING...", (w // 2 - 55, h // 2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA,
        )
    elif status in ("thinking", "acting"):
        amber_overlay = frame.copy()
        cv2.rectangle(amber_overlay, (0, 0), (w, h), (0, 100, 200), -1)
        cv2.addWeighted(amber_overlay, 0.15, frame, 0.85, 0, frame)
        label = status.upper() + "..."
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            frame, label, ((w - lw) // 2, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1, cv2.LINE_AA,
        )


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"'{MODEL_PATH}' not found. Download it from:\n"
            "  https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        )

    jarvis = JarvisAgent()
    lm = build_landmarker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    screen_w, screen_h = pyautogui.size()

    smooth_x = smooth_y = 0.0
    prev_lm9_y = 0.0
    left_down = right_down = scroll_active = False
    last_fire = {"left": 0.0, "right": 0.0}
    fist_start = None
    show_window = True
    frame_count = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        mp_img = mp_vision.Image(
            image_format=mp_vision.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = lm.detect_for_video(mp_img, int(time.time() * 1000))

        gesture_label = ""

        if result.hand_landmarks:
            lm_pts = result.hand_landmarks[0]

            # Cursor: EMA on lm9
            sx = lm_pts[9].x * screen_w
            sy = lm_pts[9].y * screen_h
            smooth_x = EMA_ALPHA * sx + (1 - EMA_ALPHA) * smooth_x
            smooth_y = EMA_ALPHA * sy + (1 - EMA_ALPHA) * smooth_y
            mouse.move(int(smooth_x), int(smooth_y), absolute=True)

            index_tip, middle_tip = resolve_index_middle(lm_pts)

            # Scroll: thumb + pinky
            d_pi = dist3d(lm_pts[4], lm_pts[20])
            scroll_active = hysteresis(d_pi, PINCH_TRIGGER, PINCH_RELEASE, scroll_active)

            if scroll_active:
                delta = (prev_lm9_y - smooth_y) * SCROLL_SCALE
                if int(delta):
                    mouse.wheel(int(delta))
                gesture_label = "SCROLL"
            else:
                # Left click: thumb + index
                d_li = dist3d(lm_pts[4], index_tip)
                new_left = hysteresis(d_li, PINCH_TRIGGER, PINCH_RELEASE, left_down)
                now = time.time()
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

                # Right click: thumb + middle
                d_rm = dist3d(lm_pts[4], middle_tip)
                new_right = hysteresis(d_rm, PINCH_TRIGGER, PINCH_RELEASE, right_down)
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

            # Fist → summon Jarvis
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
            # Hand lost — release any held buttons
            if left_down:
                mouse.release("left")
                left_down = False
            if right_down:
                mouse.release("right")
                right_down = False

        draw_hud(frame, jarvis.status, int(smooth_x), int(smooth_y), gesture_label, frame_count)

        if show_window:
            cv2.imshow("J.A.R.V.I.S.", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("t"):
            show_window = not show_window
            if not show_window:
                cv2.destroyAllWindows()
        elif key == 27:
            break

        frame_count += 1

    # Clean up
    if left_down:
        mouse.release("left")
    if right_down:
        mouse.release("right")
    if jarvis.listening:
        jarvis.stop_and_process()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
