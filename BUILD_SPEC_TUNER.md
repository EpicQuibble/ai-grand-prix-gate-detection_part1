# Build Spec: Part 1 Gate Detection (color-first pipeline + visual tuner)

This document is an implementation specification. Implement exactly what is written
here. Make no architectural decisions of your own: every parameter, default,
data structure, and behavior is specified below. If something appears
underspecified, implement the simplest reading of the text and leave a `# TODO`
comment; do not invent features. No emojis. No extra files. No narrative logging.

--------------------------------------------------------------------------------

## Part 1 goal and success criteria

Goal: reliably detect and outline racing gates in FPV arena footage (the DCL test
clip in this folder) and future similar/VQ1 footage, using pure classical computer
vision. No neural network is used in Part 1.

Context that drives the design (verified from the test footage): the gates are a
saturated purple color with a green LED strip at the base, set against a
desaturated grey arena. The only other bright objects are white ceiling lights,
which are high-brightness but low-saturation. Therefore a saturation-gated color
threshold isolates the gates almost completely. This matches the stated Virtual
Qualifier 1 conditions (desaturated environment, highlighted gates, high
signal-to-noise).

Success criteria (Part 1 is "done" when all hold on the test clip, using one saved
`gate_config.json`):

1. Every clearly visible gate (occupying at least ~4% of frame height, not at the
   extreme frame edge, not in the worst motion blur) is outlined with a 4-corner
   polygon in at least 90% of the frames in which it is visible.
2. No more than 1 false-positive gate per 20 frames on average across the clip.
3. The pipeline runs at 15 FPS or better at `proc_width = 640` on the target
   laptop (CPU only; achievable because there is no neural network).
4. The parameters that achieve this are saved in `gate_config.json` and both the
   tuner and the overlay reproduce the result from that file.

Explicitly out of scope for Part 1 (do not implement): any neural network, YOLO,
trained gate model, tracking across frames, or control/flight logic. A trained
gate detector is Part 2.

--------------------------------------------------------------------------------

## Files to produce

1. `gate_detection_core.py` — new. Pure detection logic. No UI, no windows, no
   screen capture, no threads. Imports allowed: `cv2`, `numpy`, `json`, `time`.
   (No `ultralytics`, no `torch`.)
2. `tuner.py` — new. OpenCV visual tuning tool. Imports `gate_detection_core`.
3. `gate_overlay.py` — modify. Replace its detection internals with calls to the
   core; load `gate_config.json` on startup. Keep its capture/overlay/threading.
4. `gate_config.json` — new. Written by the tuner, read by both tools. Schema in
   the config section. If missing, the core uses the built-in defaults and does
   not crash.

--------------------------------------------------------------------------------

## The pipeline (this is the whole algorithm)

For one BGR frame:

1. Resize to `proc_width` (keep aspect ratio); remember the scale factor.
2. If `preprocess_enabled`: bilateral filter
   `cv2.bilateralFilter(img, 9, bilateral_sigma, bilateral_sigma)`.
3. Convert to HSV (`cv2.COLOR_BGR2HSV`, OpenCV H is 0-180).
4. Build the color mask by OR-ing the enabled hue bands. For each enabled band X
   (A and B), `cv2.inRange(hsv, (band_X_h_lo, sat_min, val_min),
   (band_X_h_hi, 255, val_max))`. OR the band masks together with
   `cv2.bitwise_or`. If no band is enabled, the mask is all zeros.
5. Clean the mask: `MORPH_CLOSE` then `MORPH_OPEN` with an elliptical kernel of
   size `(morph_kernel, morph_kernel)`.
6. `cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`.
7. For each contour, compute candidate metrics:
   - `area = cv2.contourArea(contour)`; `area_frac = area / (proc_w * proc_h)`.
   - Reject immediately if `area_frac < min_area_frac`.
   - `rect = cv2.minAreaRect(contour)` -> `(cx, cy), (w, h), angle`.
   - `aspect = max(w, h) / max(min(w, h), 1e-6)`.
   - `hull = cv2.convexHull(contour)`; `solidity = area / max(cv2.contourArea(hull), 1e-6)`.
   - `corners = cv2.boxPoints(rect)` (4 points).
8. Validate the candidate (all must hold, else reject):
   - `min_aspect <= aspect <= max_aspect`.
   - `solidity >= min_solidity`.
9. Confidence (0..1):
   - `size_score = min(1.0, area_frac / 0.02)` (a gate at >=2% of the frame scores
     full on size).
   - `confidence = 0.5 * solidity + 0.5 * size_score`.
   - Keep the gate only if `confidence >= confidence_threshold`.
10. Build the gate result. Scale `corners`, center `(cx, cy)`, and the bounding box
    back to the ORIGINAL frame size using the scale factor from step 1.

Sort gates by `area_frac` descending (closest first). Return them.

Gate result dict keys (the overlay expects these; keep names exact):
```
x1, y1, x2, y2          # axis-aligned bbox of the corners, original-frame pixels
width, height           # bbox size
center_x, center_y      # gate center, original-frame pixels
corners                 # np.ndarray shape (4,2), original-frame pixels
confidence              # float 0..1
band                    # "A" or "B" or "A+B": which band(s) contributed most
aspect, solidity, area_frac   # metrics, for display/debug
```

--------------------------------------------------------------------------------

## gate_detection_core.py

### Class

```
class GateDetector:
    def __init__(self, config: dict | None = None)
```
Set every parameter in the config schema from `config` if given, else the default.
No I/O or windows in `__init__`.

### Config helpers

```
def to_dict(self) -> dict
def update(self, **params) -> None
def load_config(self, path) -> None          # missing file: keep defaults, no error
def save_config(self, path) -> None
@classmethod
def from_config_file(cls, path) -> "GateDetector"
```
Unknown keys in a loaded file are ignored; missing keys keep defaults.

### Primitives (each takes the already-resized BGR frame)

```
def preprocess(self, bgr) -> bgr
def color_mask(self, bgr) -> (mask uint8, band_a_mask uint8, band_b_mask uint8)
def candidates(self, mask) -> list[dict]     # steps 6-9, in proc space
```
`candidates` returns per-contour dicts with keys: `corners` (proc space),
`center`, `bbox`, `aspect`, `solidity`, `area_frac`, `confidence`, `band`,
`passed` (bool: passed validation AND confidence threshold), and `reject_reason`
(str: "" if passed, else one of "area", "aspect", "solidity", "confidence").
Determine `band` by which band mask has more overlap with the contour's filled
region; "A+B" if both are within 20% of each other and both nonzero.

### Production entry point

```
def process_frame(self, bgr) -> list[dict]
```
Runs the full pipeline, returns only gates with `passed == True`, coordinates
scaled to the ORIGINAL frame, sorted by `area_frac` desc, using the exact result
keys above.

### Debug entry point (used by the tuner)

```
def process_frame_debug(self, bgr) -> dict
```
Returns everything the tuner renders (all images in proc space unless noted):
```
{
  "proc":        BGR resized frame actually processed,
  "preprocessed":BGR,
  "hsv_h":       uint8 1ch (the H channel),
  "hsv_s":       uint8 1ch (the S channel),
  "hsv_v":       uint8 1ch (the V channel),
  "mask_a":      uint8 1ch,
  "mask_b":      uint8 1ch,
  "mask_raw":    uint8 1ch (bands OR'd, before morphology),
  "mask_clean":  uint8 1ch (after morphology),
  "candidates":  list of candidate dicts from candidates() (proc space),
  "scale":       float (proc -> original),
  "timings":     {"color_ms":float,"contours_ms":float,"total_ms":float}
}
```
No YOLO caching logic is needed because there is no neural network; the whole
pipeline is cheap, so the tuner recomputes it on every change.

--------------------------------------------------------------------------------

## tuner.py

OpenCV application. No tkinter, no matplotlib, no threads. Single blocking loop.

### Startup

- `python tuner.py [video_path]`. If no path, use the first `*.mp4` in the current
  directory; if none, print one error line and exit.
- Load `gate_config.json` if present else defaults; build one `GateDetector`.
- Open the video with `cv2.VideoCapture`; read `CAP_PROP_FRAME_COUNT`. Start at
  frame 0.

### Windows

1. `"pipeline"` — the mosaic.
2. `"gate zoom"` — the selected-candidate breakdown.
3. `"controls"` — trackbars.

### Trackbars (on "controls"; integer-only, convert as shown)

| label            | int range | -> param                 |
|------------------|-----------|--------------------------|
| frame            | 0..count-1| seek (see below)         |
| proc_width       | 0..960    | proc_width = v + 320     |
| bilateral_sigma  | 1..30     | bilateral_sigma = v      |
| band_a_h_lo      | 0..180    | band_a_h_lo = v          |
| band_a_h_hi      | 0..180    | band_a_h_hi = v          |
| band_b_h_lo      | 0..180    | band_b_h_lo = v          |
| band_b_h_hi      | 0..180    | band_b_h_hi = v          |
| sat_min          | 0..255    | sat_min = v              |
| val_min          | 0..255    | val_min = v              |
| val_max          | 0..255    | val_max = v              |
| morph_kernel     | 1..21     | morph_kernel = v (force odd; if even use v+1) |
| min_area x10000  | 1..500    | min_area_frac = v/10000  |
| min_solidity x100| 0..100    | min_solidity = v/100     |
| min_aspect x100  | 10..300   | min_aspect = v/100       |
| max_aspect x100  | 10..500   | max_aspect = v/100       |
| gate_conf x100   | 0..100    | confidence_threshold = v/100 |

Initialize each trackbar position from the current param (inverse conversion).
Every trackbar callback writes the param onto the detector and sets a `dirty` flag.
The `frame` trackbar sets `frame_idx` and flags a frame change.

### Main loop

- If a frame change is pending: `cap.set(CAP_PROP_POS_FRAMES, frame_idx)`, read the
  frame, store it.
- If `dirty` or frame changed: `result = detector.process_frame_debug(frame)`;
  rebuild and `imshow` the mosaic and zoom; clear flags.
- `cv2.waitKey(30)`; handle keys. When nothing is dirty, the loop only waits.

### Mosaic ("pipeline")

Fixed 2 rows x 4 columns, total width capped at 1280 (cell width = 320; cell height
keeps the proc frame aspect). Convert 1-channel panels to BGR before tiling. Resize
each panel to the cell size, draw a text label (FONT_HERSHEY_SIMPLEX 0.4, white
with a 1px black outline drawn first) top-left. Separate cells with a 2px black
border. Panels in order:

1. proc                          "1 original"
2. preprocessed                  "2 preprocess s=<bilateral_sigma>"
3. hsv_s                          "3 saturation"   (this is the key channel)
4. mask_a                         "4 band A H<a_lo>-<a_hi>"
5. mask_b                         "5 band B H<b_lo>-<b_hi>"
6. mask_clean                     "6 mask (morph k=<morph_kernel>)"
7. contours view                  "7 candidates"
8. annotated proc                 "8 result"

Panel 7 "candidates": proc frame copy with every candidate's `minAreaRect` polygon
drawn — green if `passed`, red if rejected — plus its `reject_reason` in small text
at the candidate center.

Panel 8 "result": proc frame copy with only passed gates drawn as polygons colored
by confidence (green >0.8, yellow >0.5, else red), the selected candidate's polygon
thickened in cyan, and its center dot.

Bottom status bar (full width, ~24px, black): `frame <idx>/<count>  passed
<n_passed>/<n_candidates>  color <color_ms>ms total <total_ms>ms  [SPACE play][a/d
step][ [ ] select ][w save cfg][s snapshot]`.

### Zoom ("gate zoom")

For the selected candidate (cycle with `[` and `]`; clamp to range; if none show a
black image with "no candidates"): left side stacks the candidate's cropped region
from `proc`, its crop from `mask_clean`, and its crop from `hsv_s`, each upscaled so
width >= 240 (INTER_NEAREST), labeled. Right side is a black text panel listing:
```
candidate <i>/<n>
band        <A|B|A+B>
area_frac   <..>   (min <min_area_frac>)
aspect      <..>   (<min_aspect>..<max_aspect>)
solidity    <..>   (min <min_solidity>)
confidence  <..>   (thr <confidence_threshold>)   colored green/yellow/red
result      PASS / REJECT (<reject_reason>)
```

### Keys

```
SPACE   play/pause (play = advance one frame per loop iteration)
d / RIGHT   +1 frame        a / LEFT   -1 frame
D           +30 frames      A          -30 frames
[ / ]       select prev/next candidate
p           toggle preprocess_enabled
1           toggle band_a_enabled
2           toggle band_b_enabled
w           save params to gate_config.json (print confirmation)
s           snapshot mosaic PNG + params JSON to ./snapshots/ (UTC timestamp)
q / ESC     quit
```
Keep the `frame` trackbar in sync when stepping (`cv2.setTrackbarPos`).

### Snapshots

Create `./snapshots/` if missing. Files: `snap_YYYYMMDD_HHMMSS_mosaic.png` and
`snap_YYYYMMDD_HHMMSS_params.json` (`detector.to_dict()` plus `"frame": idx`).

--------------------------------------------------------------------------------

## gate_overlay.py changes

- Delete the old detection methods (`preprocess_frame`, `detect_edges`,
  `detect_color`, `detect_corners`, `find_corners_active_vision`,
  `validate_gate_prior`, `find_gate_polygon`, `process_detections`) and the YOLO
  model load.
- In `__init__`: `self.detector = GateDetector.from_config_file("gate_config.json")`.
- Add a method `process_detections(self, frame)` that returns
  `self.detector.process_frame(frame)` (keep the name so the detection thread is
  unchanged).
- The overlay draws `gate['corners']` as a polygon (fall back to the bbox if
  corners is None) and uses `gate['confidence']` for color. Update the drawing code
  to read the new result keys (`confidence`, `corners`) instead of the old ones
  (`fusion_confidence`, `hull`, `methods`).
- Control-panel sliders (if kept) call `self.detector.update(param=value)`. Remove
  sliders for deleted parameters (Canny, corner sensitivity, method weights,
  active vision, gate-prior tolerance). Keep display toggles on the overlay object.
- Do not change the capture/overlay/threading code otherwise.

--------------------------------------------------------------------------------

## gate_config.json schema and defaults

```
{
  "proc_width": 640,

  "preprocess_enabled": true,
  "bilateral_sigma": 10,

  "band_a_enabled": true,
  "band_a_h_lo": 120,
  "band_a_h_hi": 160,

  "band_b_enabled": false,
  "band_b_h_lo": 40,
  "band_b_h_hi": 85,

  "sat_min": 60,
  "val_min": 50,
  "val_max": 255,

  "morph_kernel": 5,

  "min_area_frac": 0.0008,
  "min_solidity": 0.70,
  "min_aspect": 0.40,
  "max_aspect": 2.50,

  "confidence_threshold": 0.30
}
```

The purple band (A) defaults are a starting point tuned to the test footage; the
green LED band (B) is off by default. Tuning these against the real footage in the
tuner is the point of Part 1.

--------------------------------------------------------------------------------

## Acceptance checklist

- `python -c "import gate_detection_core"` imports with no error and no torch/YOLO
  dependency.
- `python tuner.py "<the test mp4>"` opens three windows and shows the 8-panel
  pipeline and zoom.
- The saturation panel (3) clearly shows the gates bright and the background dark.
- Moving `sat_min` / band hue trackbars visibly changes the masks in real time.
- `[` / `]` cycles candidates; the zoom follows and shows PASS/REJECT with reason.
- Stepping frames with a/d works and the `frame` trackbar stays in sync.
- `w` writes `gate_config.json`; re-launching the tuner restores those values.
- `s` writes a mosaic PNG and params JSON to ./snapshots/.
- `python gate_overlay.py` runs, loads `gate_config.json`, and draws gate polygons.

## Do NOT

- Do not add YOLO, torch, any neural network, tracking, or control logic.
- Do not add threads to the tuner, or matplotlib/tkinter to it.
- Do not add emojis, extra markdown files, or narrative logging.
- Do not add parameters or pipeline stages beyond those specified here.
