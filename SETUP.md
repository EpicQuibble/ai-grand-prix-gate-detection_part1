# Setup Instructions

## Quick Start

### 1. Install Dependencies (First Time Only)
```bash
cd part1
pip install -r requirements.txt
```

This installs:
- `ultralytics` - YOLOv8 (already trained model)
- `opencv-python` - Video processing
- `torch` - Deep learning
- `torchvision` - Image utilities
- `numpy` - Array operations
- `Pillow` - Image rendering for overlay
- `mss` - Fast screen capture

### 2. Run the Overlay Detector

```bash
cd part1
python gate_overlay.py
```

This will:
1. Start capturing your screen
2. Open an overlay window (stays on top)
3. Open a control panel window
4. Begin detecting gates in real-time

### 3. Before Running

Make sure to:
1. Open your video in a media player (VLC, Windows Media Player, etc.)
2. Position the video window so you can see it
3. Run gate_overlay.py
4. The overlay will appear on top of your video window

### 4. Use the Control Panel

- **Play/Pause**: Stop detection temporarily
- **Confidence Threshold**: Adjust detection sensitivity (lower = more detections, higher = fewer false positives)
- **Show Labels**: Toggle gate names and scores
- **Show Centers**: Toggle center dot visualization
- **Statistics**: Watch live gate count and gates passed

## Troubleshooting

### "No module named 'mss'"
```bash
pip install mss
```

### "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### YOLO Model Downloads Slowly
First run downloads the ~3MB model. This is normal and only happens once.

### Overlay Window Not Showing
- Make sure you have at least two monitors configured (or virtual desktop)
- Try adjusting monitor_index in the code
- Make sure your video player window is in the foreground

### Detection is Slow
- Close other applications
- Lower screen resolution if possible
- Reduce confidence threshold to skip more frames
- Use batch processor for offline analysis

## File Structure

```
part1/
├── gate_overlay.py              ← Main script (screen capture)
├── gate_detector_batch.py       ← Offline processor
├── requirements.txt             ← Python dependencies
├── README.md                    ← What it does
├── SETUP.md                     ← This file
├── WINNING_TECHNIQUES.md        ← Pipeline stages and tuning
├── BUILD_SPEC_TUNER.md          ← Spec for the visual tuner
└── logs/                        ← Created automatically at runtime
    └── gate_analysis_YYYYMMDD_HHMMSS.json
```

## What Happens When You Run It

1. Screen capture starts (invisible)
2. Overlay window appears (transparent, on top)
3. Control panel opens
4. YOLO model initializes (downloads if first time)
5. Real-time detection begins
6. Adjust settings with control panel

## Check Installation

```bash
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import torch; print('PyTorch OK')"
python -c "import mss; print('MSS OK')"
```

All should print OK

## Performance Tips

- **Fast detection?** Detection runs in background thread, doesn't block UI
- **Adjust sensitivity live**: Confidence slider changes detection in real-time
- **Want faster?** Use batch processor (offline analysis)
- **CPU usage high?** Close other apps, reduce monitor resolution temporarily

## Next: When Simulator Access Arrives

1. Run simulator
2. Run gate_overlay.py
3. Overlay will detect gates from simulator footage
4. Use control panel to tune detection parameters
5. Batch process for offline analysis and logging
