# Setup

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Verify:

```bash
python -c "import cv2; print('OpenCV OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import mss; print('MSS OK')"
```

## Run the overlay

1. Open a gate video in any media player (VLC, Windows Media Player, etc.).
2. Run:

```bash
python gate_overlay.py
```

This starts screen capture, a transparent always-on-top overlay window, and a
tkinter control panel. Detected gates are drawn over whatever the overlay is
capturing.

### Control panel

- Detection on/off
- Color thresholds (hue bands, saturation floor, value range)
- Shape validation (min area, aspect, solidity)
- Display toggles (labels, centers)
- Live statistics

For tuning the color thresholds against footage frame by frame, use the visual
tuner (`tuner.py`, see `BUILD_SPEC_TUNER.md`).

## Troubleshooting

- `No module named 'X'` — `pip install X` (e.g. `mss`, `opencv-python`).
- Overlay not visible — make sure the video player window is in the foreground;
  the overlay draws on top of the captured screen region.
- Nothing detected — the color thresholds likely do not match the gate color in
  your footage. Tune the hue bands and saturation floor in the tuner.

## Notes

Test footage (`*.mp4`) is gitignored. Supply your own video. The pipeline is pure
OpenCV; no model download is required.
