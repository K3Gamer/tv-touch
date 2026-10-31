# tv-touch

Turn a regular (non-touch) TV into a **single-point touch surface** using a webcam.

A MediaPipe hand tracker follows your index fingertip, a 4-point homography maps
the camera view to screen coordinates, and a pinch gesture injects real Windows
mouse events. It is designed for setups where the PC screen is mirrored or cast
to the TV (for example Sunshine + Moonlight), so the TV itself needs no touch
hardware.

```
webcam -> MediaPipe Hands -> index fingertip -> homography -> screen coords -> Win32 mouse
```

## Gestures

| Gesture | Action |
| --- | --- |
| Move index finger | Move the cursor |
| Pinch (thumb + index) | Click |
| Pinch twice quickly (< 400 ms, < 40 px apart) | Double-click |
| Pinch, move, release | Drag and drop |

## Requirements

- Windows 10/11
- Python 3.10 (MediaPipe wheels)
- A webcam. A phone camera shared over Iriun is a cheap and effective option.

## Install

```bat
python -m pip install -r requirements.txt
```

## Calibrate and run

```bat
python main.py --camera 0 --recalibrate
```

Calibration shows four targets. For each one: hold an **open hand** briefly,
then **pinch (thumb + index) and hold ~0.35 s** on the target. A camera preview
at the bottom of the screen shows whether the hand is detected and the current
pinch ratio. The result is saved to `calibration.json` for next time.

Next runs reuse the saved calibration:

```bat
python main.py --camera 0
```

Useful options:

```text
--list                 list camera indices and exit
--camera N             camera index (default: first working camera)
--recalibrate          force a new calibration
--preview              show a debug window (visible on the TV when mirroring)
--click pinch|dwell    click method (default: pinch)
--pinch-close 0.35     pinch ratio that presses
--pinch-open 0.55      pinch ratio that releases
--double-click-ms 400  max gap between two taps for a double-click
--double-click-dist 40 max cursor distance between two taps for a double-click
--rotate 0|90|180|270  rotate camera frames
--flip                 mirror camera frames horizontally
--margin 0.10          calibration target inset (fraction of the screen)
```

Press `Ctrl+C` to stop. With `--preview`, `q`/`Esc` quits and `r` recalibrates.

## Windows launcher

`run.bat` (not committed, create your own) can stop an old instance, delete the
old calibration and start a fresh one:

```bat
@echo off
cd /d "%~dp0"
python -u main.py --camera 0 --recalibrate
pause
```

## How double-click is made reliable

Webcam tracking jitters by a few pixels, which is enough for Windows to treat
two taps as separate clicks. When a second pinch happens within
`--double-click-ms` and `--double-click-dist`, the cursor is snapped back to the
first tap position before pressing, so the two button-down events land on the
same pixel and Windows reports a double-click.

## Notes

- `calibration.json`, logs and `__pycache__` are machine-specific and ignored.
- This project only needs a single touch point; multi-touch is not supported.
- If the cursor does not appear on the TV while casting, the streaming client
  may hide the local cursor; keep the host session visible and using the
  desktop while streaming.
