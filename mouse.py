"""Low-level Windows mouse injection (no extra dependencies)."""

import ctypes

user32 = ctypes.windll.user32
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = ctypes.c_bool
user32.mouse_event.argtypes = [
    ctypes.c_ulong,
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_ulong,
    ctypes.c_void_p,
]
user32.mouse_event.restype = None

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def enable_dpi_awareness():
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


def move(x, y):
    user32.SetCursorPos(int(round(x)), int(round(y)))


def left_down():
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, None)


def left_up():
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, None)


def click():
    left_down()
    left_up()
