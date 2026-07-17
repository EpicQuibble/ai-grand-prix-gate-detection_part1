# Part 1: Gate Detection

Detect and localize racing gates in a video stream, and provide tooling to tune
the detection pipeline by eye. This is the perception stage of the autonomy stack;
control and simulator integration come in later parts.

## Contents

- `gate_overlay.py` — real-time detection overlay. Screen-captures a video player
  window, runs the detection pipeline, and draws results in a transparent
  always-on-top window with a live tkinter control panel. Working.
- `WINNING_TECHNIQUES.md` — the detection pipeline stages and their tunable
  parameters.
- `BUILD_SPEC_TUNER.md` — specification for a frame-by-frame visual tuning tool
  (`tuner.py`) and a shared, UI-free detection core (`gate_detection_core.py`).
  Specified, not yet implemented.
- `SETUP.md` — install and run instructions.

## Detection pipeline

Modular classical-CV approach (chosen over end-to-end learning for
interpretability and low compute):

1. Preprocess — bilateral filter to reduce noise while preserving edges.
2. YOLOv8n — initial region localization.
3. Semantic segmentation — weighted fusion of three masks: Canny edges, HSV color,
   Harris corners.
4. Active-vision corner finding — sample only gate-boundary pixels.
5. Gate-prior validation — reject detections whose aspect ratio is implausible.

Note: `gate_overlay.py` uses the stock `yolov8n` COCO model, which detects general
objects rather than gates specifically. Gate-specific training is future work; for
highlighted, high-signal-to-noise gates the classical CV stages carry most of the
weight.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

With a gate video open in any media player:

```bash
python gate_overlay.py
```

See `SETUP.md` for details and troubleshooting.

## Not tracked in git

Model weights (`*.pt`, downloaded on first run) and test footage (`*.mp4`) are
gitignored. Supply your own video; the YOLO model downloads automatically.
