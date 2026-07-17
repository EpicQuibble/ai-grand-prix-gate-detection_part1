# Build Spec: Gate Detection Tuner + Core Refactor

This document is an implementation specification. Follow it exactly. Do not add
features, files, documentation, or commentary beyond what is listed here. Do not
use emojis anywhere. Do not create extra README/summary files.

The goal: split the detection logic out of the tkinter overlay into a shared,
UI-free core module, then build an OpenCV-based visual tuning tool that shows
every intermediate stage of the pipeline for one frame at a time, with live
trackbars, so parameters can be tuned by eye and saved to a config file that both
the tuner and the live overlay load.

--------------------------------------------------------------------------------

## 0. Hard constraints (read first — these drive every decision)

Target machine: Intel i7-10510U, 4 cores, no dedicated GPU (YOLO runs on CPU),
~1.7 GB free RAM, Windows 10. This is a weak machine. The following rules are
NON-NEGOTIABLE and exist because of it:

1. The tuner NEVER runs a real-time playback loop with YOLO in it. It processes
   exactly ONE frame on demand — when the user steps to a new frame or changes a
   parameter. Between those events the program sits in a blocking `cv2.waitKey`
   and does no work.

2. YOLO is the single most expensive operation. It must be cached per frame. When
   the user changes a cheap CV parameter (Canny, HSV, corner, weights, threshold),
   the tool MUST reuse the cached YOLO boxes and only recompute the cheap CV
   stages. YOLO re-runs ONLY when the frame index changes or the YOLO confidence /
   processing-width parameter changes. This is the most important performance
   decision in this spec. Get it right.

3. All processing happens at a reduced resolution controlled by `proc_width`
   (default 640). The frame is downscaled to this width (preserving aspect ratio)
   BEFORE any detection. Everything downstream works in this reduced space.

4. The mosaic display window has a fixed maximum total width (1280 px). Panels are
   sized to fit inside it. Never blit full 1080p arrays to the screen.

5. Record and display per-stage timings (milliseconds) so the user can see what is
   slow on their hardware and disable it.

--------------------------------------------------------------------------------

## 1. Files to produce

1. `gate_detection_core.py`  — new. Pure detection logic. No tkinter, no OpenCV
   windows, no screen capture, no threads. Imports allowed: `cv2`, `numpy`,
   `json`, `time`, `ultralytics.YOLO`.

2. `tuner.py` — new. The OpenCV visual tuning tool. Imports `gate_detection_core`.

3. `gate_overlay.py` — modify. Delete the detection methods that now live in the
   core; import and use the core instead. Load parameters from `gate_config.json`
   on startup. Everything else about the overlay stays as-is.

4. `gate_config.json` — new. Written by the tuner, read by both tools. Schema in
   section 5. If the file is missing, the core uses the hardcoded defaults in
   section 5 and does not crash.

--------------------------------------------------------------------------------

## 2. gate_detection_core.py

### 2.1 Class

```
class GateDetector:
    def __init__(self, config: dict | None = None, model_path: str = "yolov8n.pt")
```

- Loads the YOLO model once in `__init__`.
- Initializes every parameter in section 5 from `config` if provided, else the
  default. Store each as a plain attribute (e.g. `self.canny_low`).
- Do NOT open any window, thread, or capture device.

### 2.2 Config helpers

```
def to_dict(self) -> dict          # return all params in section 5 as a flat dict
def update(self, **params) -> None # set any subset of params by name
def load_config(self, path) -> None
def save_config(self, path) -> None
@classmethod
def from_config_file(cls, path, model_path="yolov8n.pt") -> "GateDetector"
```

`load_config` / `from_config_file`: if the file does not exist, keep defaults and
return normally (no exception). Ignore unknown keys in the file. Missing keys keep
their default.

### 2.3 Cheap CV primitives (no YOLO, operate on any BGR image)

These are ported directly from the current `gate_overlay.py`. Keep the same math.
Each takes a BGR image and returns as described. They must work on a full frame OR
a cropped gate region — the caller decides what to pass.

```
def preprocess(self, frame_bgr) -> frame_bgr
    # bilateral filter if self.preprocess_enabled else return input unchanged
    # cv2.bilateralFilter(frame, 9, self.bilateral_sigma, self.bilateral_sigma)

def edges(self, frame_bgr) -> uint8 single-channel
    # gray -> bilateralFilter(gray,5,10,10) -> Canny(canny_low,canny_high)
    # -> dilate 3x3 ellipse 1 iter

def color_mask(self, frame_bgr) -> uint8 single-channel
    # HSV inRange([h_min,s_min,v_min],[180,255,255]) -> MORPH_CLOSE then MORPH_OPEN, 5x5 ellipse

def corner_response(self, frame_bgr) -> float32 array
    # gray -> cornerHarris(2,3,0.04)

def combined_mask(self, frame_bgr) -> (uint8 binary mask, methods:list[str], confidences:dict)
    # Replicate current find_gate_polygon Stage-1 fusion exactly:
    #   combined = zeros float32
    #   if edge_enabled:   combined += edges*edge_weight;    methods+='edge';  conf['edge']=mean(edges)/255
    #   if color_enabled:  combined += color*color_weight;   methods+='color'; conf['color']=mean(color)/255
    #   if corner_enabled: cm=(corner> corner_threshold*corner.max()); combined += cm*corner_weight;
    #                      methods+='corner'; conf['corner']=mean(cm)
    #   normalize combined to 0..255 uint8
    #   threshold at self.mask_threshold  (NEW param, replaces hardcoded 50)
    #   MORPH_CLOSE 3x3 ellipse
    # Return the binary mask, methods, confidences. If no methods enabled, mask is all-zero.
```

Note: `mask_threshold` is a new parameter (default 50) replacing the hardcoded
value in the current code.

### 2.4 Per-gate geometry (ported unchanged)

```
def find_corners_active_vision(self, gate_mask, box) -> np.ndarray|None   # unchanged logic
def validate_gate_prior(self, corners, box) -> (bool, corners)            # unchanged logic
```

`box` is a tuple `(x1, y1, x2, y2)` in the reduced/proc space.

### 2.5 YOLO (the expensive call, isolated so it can be cached)

```
def yolo_boxes(self, frame_bgr) -> list[tuple[int,int,int,int,float]]
    # results = self.model(frame_bgr, conf=self.yolo_confidence, verbose=False)
    # for each box: (x1,y1,x2,y2,conf) clipped to frame bounds
    # apply class filter: if self.yolo_classes is not None, keep only those class ids
    #   (self.yolo_classes default = None meaning keep all)
    # return list
```

### 2.6 Full production pipeline

```
def process_frame(self, frame_bgr) -> list[dict]
    # 1. resize frame to proc_width (helper self._resize_to_proc, keep scale factor)
    # 2. pre = self.preprocess(resized)
    # 3. boxes = self.yolo_boxes(pre)
    # 4. for each box: region = pre[y1:y2, x1:x2]; run _gate_from_region(region, box)
    # 5. keep gates whose fused_confidence >= self.confidence_threshold and
    #    (gate_prior passes if enabled)
    # 6. scale all coordinates back to the ORIGINAL frame size before returning
    #    (so the overlay draws in original pixels)
    # Return list of gate dicts with the same keys the current overlay expects:
    #   x1,y1,x2,y2,width,height,center_x,center_y,yolo_confidence,
    #   fusion_confidence,hull,methods
```

Provide a private helper `_gate_from_region(region_bgr, box)` that runs
combined_mask -> active vision (or contour fallback) -> gate_prior and returns the
fused result (corners, methods, confidences, fused_confidence) or None. This is
the shared core of both `process_frame` and `process_frame_debug`.

### 2.7 Debug pipeline (used by the tuner)

This is the function the tuner calls. It returns EVERY intermediate as arrays so
the tuner just renders them — the tuner does no detection math itself.

```
def process_frame_debug(self, frame_bgr, cached_boxes=None) -> DebugResult
```

Behavior:
- Resize to proc_width. Record scale.
- `pre = preprocess(resized)`.
- If `cached_boxes is None`: run `yolo_boxes(pre)` and time it. Else reuse
  `cached_boxes` and record yolo time as 0.
- Compute WHOLE-FRAME stage images on `pre`: edges, color_mask, corner heatmap
  (colorize the float response with `cv2.applyColorMap` after normalizing to
  uint8, use `COLORMAP_JET`), combined_mask (return both the normalized pre-
  threshold grayscale AND the final binary).
- For each box: crop region from `pre`, compute per-gate edges/color/corner/
  combined/mask, run active vision + gate_prior, fill a GateDebug entry with all
  stats (see structure below).
- Time each stage group and put milliseconds in `timings`.

`DebugResult` and `GateDebug` may be plain dicts or dataclasses. Use this exact
key set:

```
DebugResult:
  proc_frame        # BGR, the resized frame actually processed
  preprocessed      # BGR
  edges_full        # uint8 1ch
  color_full        # uint8 1ch
  corner_heat_full  # BGR (colormapped)
  combined_full     # uint8 1ch (normalized, pre-threshold)
  mask_full         # uint8 1ch (final binary, whole frame)
  boxes             # list[(x1,y1,x2,y2,conf)]  in proc space
  gates             # list[GateDebug]
  timings           # dict: {"yolo_ms":float,"cv_ms":float,"pergate_ms":float,"total_ms":float}
  scale             # float, proc->original scale factor

GateDebug:
  box               # (x1,y1,x2,y2) proc space
  yolo_conf         # float
  region            # BGR crop
  edges             # uint8 1ch
  color             # uint8 1ch
  corner_heat       # BGR
  combined          # uint8 1ch
  mask              # uint8 1ch
  corners           # np.ndarray|None (proc space)
  methods           # list[str]
  method_conf       # dict
  fused_conf        # float
  expected_aspect   # float
  detected_aspect   # float|None
  aspect_error      # float|None
  prior_passed      # bool
  accepted          # bool  (fused_conf>=threshold AND prior ok if enabled)
```

--------------------------------------------------------------------------------

## 3. tuner.py

An OpenCV application. No tkinter. Single-threaded. Blocking event loop.

### 3.1 Startup / CLI

- `python tuner.py [video_path]`
- If no path given, search the current directory for the first `*.mp4` and use it.
  If none found, print a one-line error and exit.
- Load `gate_config.json` if present, else defaults. Build one `GateDetector`.
- Open the video with `cv2.VideoCapture`. Read total frame count
  (`CAP_PROP_FRAME_COUNT`). Start at frame 0.

### 3.2 Windows (three named windows)

1. `"pipeline"` — the mosaic (section 3.5).
2. `"gate zoom"` — the per-gate breakdown (section 3.6).
3. `"controls"` — trackbars only (section 3.3).

### 3.3 Trackbars (all on the "controls" window)

Trackbars are integer-only. Use the exact ranges and conversions below. On each
trackbar callback, write the converted value onto the detector and set the dirty
flag (section 3.4). Do NOT re-run YOLO from a CV-trackbar callback.

| Trackbar label      | int range | conversion to param            | param              | reruns YOLO? |
|---------------------|-----------|--------------------------------|--------------------|--------------|
| yolo_conf x100      | 1..90     | v/100                          | yolo_confidence    | YES          |
| proc_width          | 0..960    | v+320                          | proc_width         | YES          |
| bilateral_sigma     | 1..30     | v                              | bilateral_sigma    | YES (affects preprocess before YOLO) |
| canny_low           | 0..255    | v                              | canny_low          | no           |
| canny_high          | 0..255    | v                              | canny_high         | no           |
| hue_min             | 0..180    | v                              | color_h_min        | no           |
| sat_min             | 0..255    | v                              | color_s_min        | no           |
| val_min             | 0..255    | v                              | color_v_min        | no           |
| corner_sens x1000   | 1..50     | v/1000                         | corner_threshold   | no           |
| edge_w x10          | 0..20     | v/10                           | edge_weight        | no           |
| color_w x10         | 0..20     | v/10                           | color_weight       | no           |
| corner_w x10        | 0..20     | v/10                           | corner_weight      | no           |
| mask_thresh         | 0..255    | v                              | mask_threshold     | no           |
| min_methods         | 1..3      | v                              | min_methods        | no           |
| prior_tol x100      | 5..50     | v/100                          | gate_prior_tolerance | no         |
| search_radius       | 3..60     | v                              | corner_search_radius | no         |
| gate_conf x100      | 1..90     | v/100                          | confidence_threshold | no         |

Because `proc_width` uses `v+320`, its callback must set proc_width = trackbar+320.
Initialize every trackbar position from the current param value (inverse of the
conversion).

### 3.4 Dirty-flag state machine (the caching rule)

Maintain:
```
frame_idx           # current frame number
frame_bgr           # the raw frame read from video at frame_idx
cached_boxes        # list or None
selected_gate       # int index into result.gates
need_yolo           # bool
need_recompute      # bool
last_result         # DebugResult
```

Rules:
- Changing frame (step/seek): read new `frame_bgr`, set `cached_boxes=None`,
  `need_yolo=True`, `need_recompute=True`.
- A YES-reruns-YOLO trackbar: `cached_boxes=None`, `need_yolo=True`,
  `need_recompute=True`.
- A no-rerun trackbar: `need_recompute=True` only (keep `cached_boxes`).
- Toggling a method on/off (keys, section 3.7): `need_recompute=True` only.

Recompute step (run once per loop iteration when `need_recompute`):
```
result = detector.process_frame_debug(frame_bgr,
                                      cached_boxes=None if need_yolo else cached_boxes)
if need_yolo: cached_boxes = result.boxes
clamp selected_gate into range
last_result = result
need_yolo = False
need_recompute = False
rebuild both mosaic and zoom images and imshow them
```

If `need_recompute` is false, the loop only calls `cv2.waitKey` and handles keys.
This keeps the CPU idle when nothing is changing.

### 3.5 Mosaic builder ("pipeline" window)

Build a 2-row by 4-column grid of 8 panels from `last_result`. Panels, in order:

1. proc_frame                 label "1 original"
2. preprocessed               label "2 preprocess sigma=<bilateral_sigma>"
3. edges_full (to BGR)        label "3 canny <low>/<high>"
4. color_full (to BGR)        label "4 color H>=<h> S>=<s> V>=<v>"
5. corner_heat_full           label "5 harris <corner_threshold>"
6. combined_full (to BGR)     label "6 combined"
7. mask_full (to BGR)         label "7 mask thr=<mask_threshold>"
8. annotated proc_frame       label "8 result"

Panel 8 = copy of proc_frame with, drawn on top:
- every YOLO box as a thin gray rectangle,
- each accepted gate polygon (green if fused_conf>0.8, yellow if >0.5, else red),
- the selected gate's box highlighted in a distinct color (cyan) and thicker,
- corner points of the selected gate as small filled circles.

Sizing: each cell width = floor(1280/4) = 320. Cell height keeps the proc frame's
aspect ratio. Convert every single-channel panel to BGR with
`cv2.cvtColor(x, COLOR_GRAY2BGR)` before tiling. Resize every panel to the cell
size. Burn the label with `cv2.putText` (FONT_HERSHEY_SIMPLEX, 0.4, white, with a
1px black outline drawn first for legibility) at the top-left of each cell.
Concatenate with `np.hstack` per row then `np.vstack` the two rows. Add a thin
(2px) black separator between cells by padding panels before stacking.

Bottom status bar (one extra row, full width, ~24px tall, black background):
`frame <idx>/<total>  gates <accepted>/<total boxes>  yolo <yolo_ms>ms cv <cv_ms>ms
total <total_ms>ms  [SPACE play] [a/d step] [w save cfg] [s snapshot]`.

### 3.6 Zoom builder ("gate zoom" window)

For `last_result.gates[selected_gate]` (if any gates exist; else show a black
image with text "no gates"):

Left side: 5 stacked panels, each upscaled so the gate region is at least 240 px
wide (use `INTER_NEAREST` so pixels are crisp): region, edges, color, corner_heat,
mask. Label each.

Right side: a text panel (black background, white text) listing:
```
gate <selected>/<count>
yolo_conf   <..>
methods     <edge,color,corner>
edge_conf   <..>
color_conf  <..>
corner_conf <..>
fused_conf  <..>   (color the number green/yellow/red by the same thresholds)
exp_aspect  <..>
det_aspect  <..>
aspect_err  <..>   tol=<gate_prior_tolerance>
prior       PASS/FAIL
accepted    YES/NO
```

### 3.7 Keyboard controls (handle in the main loop via cv2.waitKey)

```
SPACE  toggle play/pause. Play mode = auto-advance one frame per loop with a
       ~30 ms wait. (Still one-frame-at-a-time processing; this is a slow
       preview, acceptable because it just steps frames. If YOLO makes it crawl,
       that is expected on this hardware — that is why step controls exist.)
d / RIGHT   step +1 frame
a / LEFT    step -1 frame
D           step +30 frames
A           step -30 frames
[ / ]       select previous / next gate
e           toggle edge_enabled
c           toggle color_enabled
r           toggle corner_enabled  (r = coRner; c is taken by color)
v           toggle active_vision_enabled
g           toggle gate_prior_enabled
p           toggle preprocess_enabled
w           save current params to gate_config.json (print confirmation)
s           snapshot: save the mosaic PNG and current params JSON into
            ./snapshots/ with a UTC timestamp filename
q / ESC     quit
```

Also add a "seek" trackbar labeled `frame` on the "controls" window, range
0..(total-1). Moving it sets frame_idx and triggers a frame change. Keep it in
sync when stepping via keys (`cv2.setTrackbarPos`).

### 3.8 Snapshot output

`./snapshots/` created if missing. Filenames:
`snap_YYYYMMDD_HHMMSS_mosaic.png` and `snap_YYYYMMDD_HHMMSS_params.json`.
The JSON is `detector.to_dict()` plus a `"frame"` key with the frame index.

--------------------------------------------------------------------------------

## 4. gate_overlay.py changes

- Remove the detection methods now in the core: `preprocess_frame`,
  `detect_edges`, `detect_color`, `detect_corners`, `find_corners_active_vision`,
  `validate_gate_prior`, `find_gate_polygon`, `process_detections`.
- In `__init__`, build `self.detector = GateDetector.from_config_file("gate_config.json")`.
- Replace the body of `process_detections(frame)` with
  `return self.detector.process_frame(frame)`. Keep the method name so the
  detection thread is unchanged.
- The control-panel tkinter sliders may stay, but each slider callback now writes
  to `self.detector` (e.g. `self.detector.update(canny_low=int(v))`) instead of
  to `self`. Keep the overlay's own display-only flags (show_labels, etc.) on the
  overlay object.
- Do not change the overlay/capture/threading code otherwise.

--------------------------------------------------------------------------------

## 5. gate_config.json schema and defaults

The core defines these defaults. The file, when present, overrides any subset.

```
{
  "proc_width": 640,
  "yolo_confidence": 0.35,
  "yolo_classes": null,

  "preprocess_enabled": true,
  "bilateral_sigma": 10,

  "edge_enabled": true,
  "canny_low": 40,
  "canny_high": 120,
  "edge_weight": 1.0,

  "color_enabled": true,
  "color_h_min": 0,
  "color_s_min": 80,
  "color_v_min": 120,
  "color_weight": 1.2,

  "corner_enabled": true,
  "corner_threshold": 0.008,
  "corner_weight": 0.8,

  "mask_threshold": 50,
  "min_methods": 1,
  "confidence_threshold": 0.25,

  "active_vision_enabled": true,
  "corner_search_radius": 15,

  "gate_prior_enabled": true,
  "gate_prior_tolerance": 0.35
}
```

--------------------------------------------------------------------------------

## 6. Acceptance checklist

- `python -c "import gate_detection_core"` imports with no error.
- `python tuner.py "<the test mp4>"` opens three windows, shows a frame's full
  8-panel pipeline, and the zoom window.
- Moving a CV trackbar (e.g. canny_low) updates the mosaic WITHOUT a visible YOLO
  recompute (yolo_ms shows 0 in the status bar after the first frame load).
- Moving yolo_conf or proc_width DOES re-run YOLO (yolo_ms > 0).
- Stepping frames with a/d works; the seek trackbar stays in sync.
- `[`/`]` cycles the selected gate and the zoom window follows.
- `w` writes gate_config.json; re-running the tuner picks those values up.
- `s` writes a mosaic PNG and params JSON into ./snapshots/.
- `python gate_overlay.py` still runs and now reads gate_config.json.

## 7. Do NOT

- Do not add threads to the tuner.
- Do not run YOLO on every trackbar move.
- Do not use matplotlib or tkinter in the tuner.
- Do not add emojis, extra markdown files, or narrative logging.
- Do not change the detection MATH when porting; copy it verbatim, only relocating
  it and adding the two new parameters (`proc_width`, `mask_threshold`).
