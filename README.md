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

Pure classical CV, color-first — no neural network in Part 1. The gates are a
saturated color (purple, with a green LED base) against a desaturated arena, so a
saturation-gated color threshold isolates them almost completely.

1. Resize to a fixed processing width; optional bilateral filter for motion blur.
2. HSV color threshold on the gate hue band(s), gated by a saturation floor that
   rejects the grey background and white ceiling lights.
3. Morphological close/open to consolidate the gate outline.
4. Contour extraction, then `minAreaRect` for the four gate corners.
5. Shape/size validation (aspect ratio, solidity, minimum area) to reject
   non-gate blobs, producing a confidence score.

A trained gate detector is deliberately left to Part 2, when labeled gate data is
available. See `BUILD_SPEC_TUNER.md` for the full specification and rationale.

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

Test footage (`*.mp4`) is gitignored — supply your own video. The pipeline is pure
OpenCV, so no model weights are needed. (`*.pt` files are also gitignored in case
Part 2 introduces a trained model.)
