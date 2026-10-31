"""Hand tracking, pinch detection and cursor smoothing."""

import math

import cv2
import mediapipe as mp
import numpy as np

INDEX_TIP = 8
THUMB_TIP = 4
WRIST = 0
MIDDLE_MCP = 9


class HandResult:
    __slots__ = ("landmarks", "index_tip", "pinch_ratio")

    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.index_tip = (float(landmarks[INDEX_TIP, 0]), float(landmarks[INDEX_TIP, 1]))
        palm = float(np.linalg.norm(landmarks[WRIST] - landmarks[MIDDLE_MCP]))
        if palm < 1e-6:
            palm = 1e-6
        pinch = float(np.linalg.norm(landmarks[THUMB_TIP] - landmarks[INDEX_TIP]))
        self.pinch_ratio = pinch / palm


class HandTracker:
    def __init__(self, max_hands=1, det_conf=0.6, track_conf=0.6, model_complexity=0):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        lm = result.multi_hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lm.landmark], dtype=np.float32)
        return HandResult(pts)

    def close(self):
        self._hands.close()


class PinchDetector:
    """Hysteresis state machine: closed when ratio <= close, open when ratio >= open."""

    def __init__(self, close=0.35, open_=0.55):
        if open_ <= close:
            raise ValueError("pinch open threshold must be greater than close threshold")
        self.close = close
        self.open = open_
        self.state = False

    def update(self, ratio):
        if ratio is None:
            self.state = False
            return False
        if not self.state and ratio <= self.close:
            self.state = True
        elif self.state and ratio >= self.open:
            self.state = False
        return self.state

    def reset(self):
        self.state = False


class OneEuro:
    """1D One Euro filter: low jitter at rest, low lag when moving fast."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._raw_prev = None
        self._filt_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None:
            self._raw_prev = x
            self._filt_prev = x
            self._t_prev = t
            return x
        dt = max(t - self._t_prev, 1e-4)
        dx = (x - self._raw_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._filt_prev
        self._raw_prev = x
        self._filt_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat
