# Setup

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Verify:

```bash
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import mss; print('MSS OK')"
```

The YOLO weights download automatically on first run.

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
- Per-stage parameters (YOLO confidence, Canny, HSV, corner, weights, gate-prior)
- Display toggles (labels, centers, method tags)
- Live statistics

## Troubleshooting

- `No module named 'X'` — `pip install X` (e.g. `mss`, `ultralytics`,
  `opencv-python`).
- Overlay not visible — make sure the video player window is in the foreground;
  the overlay draws on top of the captured screen region.
- Slow detection — this pipeline runs YOLO on the CPU. Close other apps, lower the
  capture resolution, or raise the YOLO confidence to process fewer boxes.

## Notes

Model weights (`*.pt`) and test footage (`*.mp4`) are gitignored. Supply your own
video; the model downloads on first run.
