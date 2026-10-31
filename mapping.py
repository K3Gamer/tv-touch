"""Geometry helpers: homography and calibration persistence."""

import json

import cv2
import numpy as np


def compute_homography(cam_points, screen_points):
    src = np.asarray(cam_points, dtype=np.float32)
    dst = np.asarray(screen_points, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.all(np.isfinite(H)):
        raise RuntimeError("Calibration failed: could not compute a valid homography.")
    return H


def apply_homography(H, point):
    v = H @ np.array([float(point[0]), float(point[1]), 1.0])
    if abs(v[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(v[0] / v[2]), float(v[1] / v[2])


def save_calibration(path, H, screen_size, camera_index):
    data = {
        "homography": H.tolist(),
        "screen_w": int(screen_size[0]),
        "screen_h": int(screen_size[1]),
        "camera_index": int(camera_index),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    H = np.array(data["homography"], dtype=np.float64)
    return H, int(data["screen_w"]), int(data["screen_h"])
