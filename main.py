"""tv-touch: use a webcam to turn a cast TV into a single-point touch surface.

Pipeline: webcam -> MediaPipe hand landmarks -> index fingertip -> homography ->
screen coordinates -> Windows mouse injection. A pinch (thumb + index) acts as a
click, so a quick pinch is a tap and holding it while moving is a drag.
"""

import argparse
import ctypes
import math
import os
import time

import cv2
import numpy as np

import mouse
from mapping import (
    apply_homography,
    compute_homography,
    load_calibration,
    save_calibration,
)
from tracker import HandTracker, OneEuro, PinchDetector

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def get_screen_size():
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def apply_view(frame, rotate, flip):
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame


def open_camera(index=None):
    indices = [index] if index is not None else list(range(0, 6))
    for i in indices:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                print(f"[camera] using index {i}")
                return cap, i
        cap.release()
    raise RuntimeError("No working camera found. Try --list or --camera N.")


def list_cameras():
    print("Scanning camera indices 0..7 (DirectShow)...")
    for i in range(8):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                print(f"  [{i}] {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  [{i}] opened but returned no frame")
        cap.release()


def calibrate(cap, tracker, screen_w, screen_h, args):
    m = args.margin
    targets = [
        (m * screen_w, m * screen_h),
        ((1 - m) * screen_w, m * screen_h),
        ((1 - m) * screen_w, (1 - m) * screen_h),
        (m * screen_w, (1 - m) * screen_h),
    ]
    win = "TV-Touch Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, screen_w, screen_h)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pinch = PinchDetector(args.pinch_close, args.pinch_open)
    cam_points = []
    start_t = time.time()
    arm_delay = 2.5
    min_open = 0.35
    hold_s = 0.35
    cooldown_s = 0.30
    last_capture = -1e9
    closed_since = None
    open_since = None
    print("[calib] Reach to the TV and PINCH (thumb + index) on the highlighted target.")
    print("[calib] Capture needs an open hand, then a steady pinch for ~0.35s.")

    def draw_inset(canvas, cam_frame, hand):
        fh, fw = cam_frame.shape[:2]
        iw = min(360, screen_w // 4)
        ih = int(iw * fh / fw)
        x0 = (screen_w - iw) // 2
        y0 = screen_h - ih - 40
        small = cv2.resize(cam_frame, (iw, ih))
        sx, sy = iw / fw, ih / fh
        if hand is not None:
            for pt in hand.landmarks:
                cv2.circle(small, (int(pt[0] * sx), int(pt[1] * sy)), 2, (0, 255, 0), -1)
            ix, iy = hand.index_tip
            cv2.circle(small, (int(ix * sx), int(iy * sy)), 7, (0, 0, 255), 2)
        canvas[y0:y0 + ih, x0:x0 + iw] = small
        cv2.rectangle(canvas, (x0, y0), (x0 + iw, y0 + ih), (70, 70, 70), 2)
        label = "camera: hand OK" if hand is not None else "camera: NO HAND"
        color = (0, 255, 0) if hand is not None else (0, 0, 255)
        cv2.putText(canvas, label, (x0, y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    while len(cam_points) < 4:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = apply_view(frame, args.rotate, args.flip)
        hand = tracker.process(frame)
        ratio = hand.pinch_ratio if hand is not None else None
        closed = pinch.update(ratio)
        now = time.time()

        if hand is None:
            closed_since = None
            open_since = None
        elif not closed:
            closed_since = None
            open_since = now
        else:
            if open_since is not None:
                if closed_since is None:
                    closed_since = now
                can_capture = (
                    now - start_t >= arm_delay
                    and now - open_since >= min_open
                    and now - closed_since >= hold_s
                    and now - last_capture >= cooldown_s
                )
                if can_capture:
                    cam_points.append(hand.index_tip)
                    last_capture = now
                    closed_since = None
                    open_since = None
                    print(f"[calib] captured point {len(cam_points)}/4 (ratio={ratio:.2f})")

        canvas = np.zeros((screen_h, screen_w, 3), np.uint8)
        for i, t in enumerate(targets):
            color = (0, 200, 0) if i < len(cam_points) else (90, 90, 90)
            cv2.circle(canvas, (int(t[0]), int(t[1])), 16, color, 2)
            cv2.putText(canvas, str(i + 1), (int(t[0]) + 24, int(t[1]) - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3, cv2.LINE_AA)
        if len(cam_points) < 4:
            t = targets[len(cam_points)]
            cv2.circle(canvas, (int(t[0]), int(t[1])), 46, (0, 255, 255), 3)
            cv2.line(canvas, (int(t[0]) - 70, int(t[1])),
                     (int(t[0]) + 70, int(t[1])), (0, 255, 255), 2)
            cv2.line(canvas, (int(t[0]), int(t[1]) - 70),
                     (int(t[0]), int(t[1]) + 70), (0, 255, 255), 2)
        if now - start_t < arm_delay:
            cv2.putText(canvas, f"Get ready... {max(arm_delay - (now - start_t), 0):.0f}",
                        (int(screen_w * 0.04), int(screen_h * 0.5) - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)
            hint = "Calibration starts in a moment. Keep your hand ready."
        else:
            hint = "Open hand, then pinch on the yellow target and hold briefly."
        cv2.putText(canvas, f"Step {min(len(cam_points) + 1, 4)}/4: {hint}",
                    (int(screen_w * 0.04), int(screen_h * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Pinch = thumb + index together.   ESC = cancel",
                    (int(screen_w * 0.04), int(screen_h * 0.5) + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        if hand is None:
            status, color = "hand: not detected", (0, 0, 255)
        elif closed:
            status, color = f"hand: OK   pinch: {hand.pinch_ratio:.2f}  (PINCH)", (0, 255, 0)
        else:
            status, color = f"hand: OK   pinch: {hand.pinch_ratio:.2f}  (open)", (0, 200, 255)
        cv2.putText(canvas, status, (int(screen_w * 0.04), int(screen_h * 0.5) + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        draw_inset(canvas, frame, hand)

        cv2.imshow(win, canvas)
        if (cv2.waitKey(1) & 0xFF) == 27:
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    return compute_homography(cam_points, targets)


def draw_preview(frame, hand, cursor, down):
    vis = frame.copy()
    if hand is not None:
        for pt in hand.landmarks:
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
        ix, iy = hand.index_tip
        cv2.circle(vis, (int(ix), int(iy)), 8, (0, 0, 255), 2)
        cv2.putText(vis, f"pinch {hand.pinch_ratio:.2f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if cursor is not None:
        cv2.putText(vis, f"cursor {cursor[0]:.0f},{cursor[1]:.0f} down={down}", (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("tv-touch", vis)


def run(cap, tracker, H, screen_w, screen_h, args):
    pinch = PinchDetector(args.pinch_close, args.pinch_open)
    fx = OneEuro(min_cutoff=args.smooth_min_cutoff, beta=args.smooth_beta)
    fy = OneEuro(min_cutoff=args.smooth_min_cutoff, beta=args.smooth_beta)

    down = False
    closed_since = None
    last_pt = None
    anchor = None
    anchor_t = 0.0
    cooldown_until = 0.0
    last_tap_t = -1e9
    last_tap_pos = None
    press_t = 0.0
    press_pos = None
    snapped = False

    if args.preview:
        cv2.namedWindow("tv-touch", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tv-touch", 720, 405)

    print("[run] Control the cursor with your index finger. Pinch to click, "
          "pinch twice quickly to double-click, hold to drag. Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame = apply_view(frame, args.rotate, args.flip)
            now = time.time()
            hand = tracker.process(frame)

            if hand is not None:
                px, py = apply_homography(H, hand.index_tip)
                if math.isfinite(px) and math.isfinite(py):
                    px = min(max(px, 0.0), screen_w - 1.0)
                    py = min(max(py, 0.0), screen_h - 1.0)
                    px = fx(px, now)
                    py = fy(py, now)
                    mouse.move(px, py)
                    last_pt = (px, py)

                    if args.click == "pinch":
                        closed = pinch.update(hand.pinch_ratio)
                        if closed:
                            if closed_since is None:
                                closed_since = now
                            if not down and now - closed_since >= 0.08:
                                snapped = False
                                if (last_tap_pos is not None
                                        and now - last_tap_t <= args.double_click_ms / 1000.0
                                        and math.hypot(px - last_tap_pos[0], py - last_tap_pos[1])
                                        <= args.double_click_dist):
                                    mouse.move(last_tap_pos[0], last_tap_pos[1])
                                    px, py = last_tap_pos
                                    last_pt = (px, py)
                                    last_tap_pos = None
                                    snapped = True
                                    print("[run] double-click")
                                mouse.left_down()
                                down = True
                                press_t = now
                                press_pos = (px, py)
                        else:
                            closed_since = None
                            if down:
                                mouse.left_up()
                                down = False
                                if (not snapped and press_pos is not None
                                        and math.hypot(px - press_pos[0], py - press_pos[1]) <= 20.0):
                                    last_tap_t = press_t
                                    last_tap_pos = press_pos
                                snapped = False
                    else:
                        if anchor is None or math.hypot(px - anchor[0], py - anchor[1]) > args.dwell_radius:
                            anchor = (px, py)
                            anchor_t = now
                        if (not down and anchor is not None and now >= cooldown_until
                                and now - anchor_t >= args.dwell_ms / 1000.0):
                            mouse.click()
                            anchor = None
                            cooldown_until = now + 0.6
            else:
                if down:
                    mouse.left_up()
                    down = False
                pinch.reset()
                closed_since = None
                anchor = None
                last_pt = None
                press_pos = None
                snapped = False

            if args.preview:
                draw_preview(frame, hand, last_pt, down)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    return "recalibrate"
    finally:
        if down:
            mouse.left_up()
        if args.preview:
            cv2.destroyAllWindows()
    return "quit"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Turn a cast TV into a single-point touch surface using a webcam.")
    parser.add_argument("--camera", type=int, default=None, help="camera index (see --list)")
    parser.add_argument("--list", action="store_true", help="list available camera indices and exit")
    parser.add_argument("--recalibrate", action="store_true", help="force a new calibration")
    parser.add_argument("--click", choices=["pinch", "dwell"], default="pinch",
                        help="click method: pinch (default) or dwell (hover to click)")
    parser.add_argument("--dwell-ms", type=int, default=900, help="dwell click delay in ms")
    parser.add_argument("--dwell-radius", type=float, default=25.0, help="dwell movement tolerance in px")
    parser.add_argument("--pinch-close", type=float, default=0.35, help="pinch ratio to press")
    parser.add_argument("--pinch-open", type=float, default=0.55, help="pinch ratio to release")
    parser.add_argument("--double-click-ms", type=int, default=400,
                        help="max time (ms) between two taps to count as a double-click")
    parser.add_argument("--double-click-dist", type=float, default=40.0,
                        help="max cursor distance (px) between two taps for a double-click")
    parser.add_argument("--smooth-min-cutoff", type=float, default=1.0, help="One Euro min cutoff")
    parser.add_argument("--smooth-beta", type=float, default=0.02, help="One Euro beta (speed response)")
    parser.add_argument("--margin", type=float, default=0.10, help="calibration target inset (fraction)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0,
                        help="rotate camera frames before processing")
    parser.add_argument("--flip", action="store_true", help="mirror camera frames horizontally")
    parser.add_argument("--preview", action="store_true",
                        help="show camera debug window (appears on TV if mirroring full desktop)")
    return parser


def main():
    args = build_parser().parse_args()
    mouse.enable_dpi_awareness()

    if args.list:
        list_cameras()
        return

    screen_w, screen_h = get_screen_size()
    print(f"[screen] {screen_w}x{screen_h}")

    cap, cam_index = open_camera(args.camera)
    tracker = HandTracker(model_complexity=0)
    try:
        while True:
            H = None
            if not args.recalibrate and os.path.exists(CALIB_PATH):
                try:
                    H, saved_w, saved_h = load_calibration(CALIB_PATH)
                    if (saved_w, saved_h) != (screen_w, screen_h):
                        print("[calib] saved calibration is for a different screen size; recalibrating.")
                        H = None
                except Exception as exc:
                    print(f"[calib] could not load calibration: {exc}")
                    H = None

            if H is None:
                H = calibrate(cap, tracker, screen_w, screen_h, args)
                if H is None:
                    print("Calibration cancelled.")
                    return
                save_calibration(CALIB_PATH, H, (screen_w, screen_h), cam_index)
                print(f"[calib] saved to {CALIB_PATH}")

            action = run(cap, tracker, H, screen_w, screen_h, args)
            if action != "recalibrate":
                break
            args.recalibrate = True
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
