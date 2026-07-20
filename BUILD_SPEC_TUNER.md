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
4. The parameters that achieve this are saved in `gate_config.json`, and the tuner reproduces the result from that file.

Explicitly out of scope for Part 1 (do not implement): any neural network, YOLO,
trained gate model, PnP / pose estimation, tracking across frames, state
estimation, or control/flight logic. Part 1 detects and outlines gates and
produces their corner points; consuming those corners (PnP for drone localization)
and replacing the color mask with a trained segmentation network are Part 2.

--------------------------------------------------------------------------------

## Files to produce

1. `gate_detection_core.py` — new. Pure detection logic. No UI, no windows, no screen capture, no threads. Imports allowed: `cv2`, `numpy`, `json`, `time`. (No `ultralytics`, no `torch`.)
2. `tuner.py` — new. OpenCV visual tuning tool. Imports `gate_detection_core`.
3. `gate_config.json` — new. Written by the tuner, read by the core on startup. Schema in the config section. If missing, the core uses the built-in defaults and does not crash.

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
6. Find contours WITH hierarchy so gate windows are available:
   `contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)`.
   RETR_CCOMP gives two levels: outer boundaries (level 0) and holes (level 1). A
   gate frame is a colored ring, so its outer edge is a level-0 contour and its
   window is that contour's level-1 hole.
7. For each LEVEL-0 contour, compute candidate metrics:
   - `area = cv2.contourArea(contour)`; `area_frac = area / (proc_w * proc_h)`.
   - Reject immediately if `area_frac < min_area_frac`.
   - `rect = cv2.minAreaRect(contour)` -> `(cx, cy), (w, h), angle`.
   - `aspect = max(w, h) / max(min(w, h), 1e-6)`.
   - `hull = cv2.convexHull(contour)`; `solidity = area / max(cv2.contourArea(hull), 1e-6)`.
8. Corner extraction (the important stage — these corners are what a future PnP
   pose solver will consume, so find the TRUE gate quadrilateral, not a box):
   - Outer corners: `peri = cv2.arcLength(contour, True)`;
     `approx = cv2.approxPolyDP(contour, approx_epsilon_frac * peri, True)`.
     If `len(approx) == 4`: `outer_corners = order_corners(approx.reshape(4, 2))`.
     Otherwise fall back to `outer_corners = order_corners(cv2.boxPoints(rect))`.
   - Inner corners (only if `find_inner_corners`): using `hierarchy`, take this
     contour's largest child hole (a level-1 contour whose parent is this one),
     approxPolyDP it the same way; if it yields exactly 4 points set
     `inner_corners = order_corners(...)`, else `inner_corners = None`.
   - `order_corners(pts)` returns the 4 points in the fixed order
     [top-left, top-right, bottom-right, bottom-left]: sort the 4 points by y; of
     the two smallest-y (top) order by x -> TL, TR; of the two largest-y (bottom)
     order by x -> BL, BR; return [TL, TR, BR, BL]. A fixed order is required so PnP
     can match image corners to known 3D corners later.
9. Validate the candidate (all must hold, else reject):
   - `min_aspect <= aspect <= max_aspect`.
   - `solidity >= min_solidity`.
10. Confidence (0..1):
   - `size_score = min(1.0, area_frac / 0.02)` (a gate at >=2% of the frame scores
     full on size).
   - `confidence = 0.5 * solidity + 0.5 * size_score`.
   - Keep the gate only if `confidence >= confidence_threshold`.
11. Build the gate result. Scale `outer_corners`, `inner_corners` (if any), center
    `(cx, cy)`, and the bounding box back to the ORIGINAL frame size using the scale
    factor from step 1.

Sort gates by `area_frac` descending (closest first). Return them.

Gate result dict keys (keep names exact; the core and tuner consume these):
```
x1, y1, x2, y2          # axis-aligned bbox of the corners, original-frame pixels
width, height           # bbox size
center_x, center_y      # gate center, original-frame pixels
corners                 # np.ndarray (4,2) outer corners, ordered TL,TR,BR,BL, original px
inner_corners           # np.ndarray (4,2) ordered, or None
confidence              # float 0..1
band                    # "A" or "B" or "A+B": which band(s) contributed most
aspect, solidity, area_frac   # metrics, for display/debug
```
The `corners` (outer) plus `inner_corners` are the up-to-8 image points a Part 2
PnP stage will pair with the known 3D gate corners to localize the drone. Part 1
only produces and validates them; it does not solve PnP.

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
def order_corners(self, pts) -> np.ndarray   # 4 points -> [TL,TR,BR,BL]
def candidates(self, mask) -> list[dict]     # steps 6-10, in proc space
```
`candidates` returns per-contour dicts with keys: `outer_corners` (proc space, 4x2
ordered), `inner_corners` (4x2 ordered or None), `center`, `bbox`, `aspect`,
`solidity`, `area_frac`, `confidence`, `band`, `passed` (bool: passed validation
AND confidence threshold), and `reject_reason` (str: "" if passed, else one of
"area", "aspect", "solidity", "confidence"). Determine `band` by which band mask
has more overlap with the contour's filled region; "A+B" if both are within 20% of
each other and both nonzero.

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
| approx_eps x1000 | 5..100    | approx_epsilon_frac = v/1000 |
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
from `proc` (with its outer corners drawn as green dots and inner corners as cyan
dots, numbered 0-3 in TL,TR,BR,BL order), its crop from `mask_clean`, and its crop
from `hsv_s`, each upscaled so width >= 240 (INTER_NEAREST), labeled. Right side is
a black text panel listing:
```
candidate <i>/<n>
band        <A|B|A+B>
corners     outer 4 / inner <4|none>
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

  "approx_epsilon_frac": 0.03,
  "find_inner_corners": true,

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

## Do NOT

- Do not add YOLO, torch, any neural network, tracking, or control logic.
- Do not add threads to the tuner, or matplotlib/tkinter to it.
- Do not add emojis, extra markdown files, or narrative logging.
- Do not add parameters or pipeline stages beyond those specified here (this constraint governs the v1 build above; the roadmap below deliberately extends the parameter, pipeline, and tuner-window set, and supersedes this line for v1.1).

--------------------------------------------------------------------------------

# Roadmap v1.1 — pipeline and tuner improvements

Everything above describes v1, which is built, committed, and running. This roadmap is the next round, driven by reviewing the live tuner against the test footage. Each task is self-contained: a diagnosis, the exact change to make, an acceptance test, and the files it touches, so we can implement them one at a time and in isolation. These tasks deliberately add parameters and tuner/pipeline stages beyond v1; where they conflict with a v1 instruction, the roadmap wins. All other v1 constraints still hold: classical CV only, OpenCV-only tuner, no threads/tkinter/matplotlib in the tuner, no neural network, no emojis, no extra files, no narrative logging. Note also that the tuner already has four windows — `pipeline`, `gate zoom`, `tune: color`, `tune: shape` — after the committed control-panel split; that supersedes the single-`controls`-window description in the v1 "Windows" and "Trackbars" sections above, and the two control panels are where the new slider behaviors (Tasks 6 and 7) apply.

## Diagnosis: is the pipeline order right, and why does tuning feel blind?

Is the stage order correct? Yes. resize -> denoise -> HSV threshold -> morphology -> contour extraction -> geometry/validation -> confidence is the standard, correct order; do not reorder it. The problems are not the order, they are three specific weaknesses, each mapped to a task below.

First, blind tuning (the "we apply the other filters over them so we cannot see what is changing" concern): the color mask is the pixelwise AND of three independent thresholds — the hue band, the `sat_min` floor, and the `val_min..val_max` window — but the tuner only ever shows the finished band mask plus the raw saturation channel. You cannot see what `sat_min` alone removes versus what the hue band alone removes, and the hue channel is never displayed at all, so setting the hue band is guesswork. This is addressed by Task 5 (per-constraint views).

Second, a filter that is quietly destroying signal (morphology): `morph_kernel` is a single kernel used for both the close and the open, and at the tuned value (k around 17 on a ~560px-wide frame) it is large enough to round the square gate outline into a blob — which wrecks corner extraction — and to seal the gate's inner window — which is why `inner_corners` comes back "none." Addressed by Task 2 (split and shrink the morphology).

Third, a filter that is missing the right signal (confidence): confidence folds raw size into the score (`0.5*solidity + 0.5*size`), so a sharp but distant gate is rejected purely for being small, while a solid round floor reflection of the same size scores the same as a hollow gate. The discriminator we are not using is hollowness: a real gate is a ring with an inner window, whereas most false positives are solid blobs. Addressed by Task 3 (score on hollowness, not raw size), which depends on Task 2 first restoring the inner hole.

## Suggested order and dependencies

Do Task 0 (git archival) first and on its own. Then the core-detection fixes in order: Task 1 (candidate ordering), then Task 2 (morphology) before Task 3 (hollowness scoring) and Task 4 (corner refinement), because Tasks 3 and 4 both need the cleaner contour and the restored inner hole that Task 2 produces. Then the tuner work: Task 5 (per-constraint views), Task 6 (undo), Task 7 (arrow-key sliders). Then Task 8 (zoom overhaul), which consumes Task 1's ordering and Task 4's corners. Then Task 9 (README) last, so it documents the final behavior. Task 10 (overlay purge) is independent and can go anytime.

--------------------------------------------------------------------------------

## Task 0 — Archive v1 as `v1_deprecated`, keep clean work on `main`

Problem: v1 lives on `main` and is pushed to origin. Before layering the cleanup and the v1.1 changes onto `main`, we want the exact v1 state preserved under an archived branch so a reviewer can always see the original.

Change (recommended, non-destructive — no history rewrite): branch `v1_deprecated` from the current v1 tip *before* committing any of the in-progress cleanup, so it captures pure v1. At the time of writing, `HEAD` (commit `66d1a57`) is still pure v1, and the working tree holds the uncommitted cleanup (README rewrite, `SETUP.md` removal, `gate_overlay.py` removal, snapshot images, `.gitignore` edits). Sequence: (1) hold the working tree (`git stash -u`) so the archive point is clean, (2) `git branch v1_deprecated HEAD`, (3) `git push -u origin v1_deprecated`, (4) `git stash pop` to bring the cleanup back, (5) commit the cleanup on `main` and `git push origin main`. `main` keeps its history and its link to origin; `main` going forward is the clean version.

Alternative (only if the user wants `main` to contain no v1 history at all): create an orphan branch (`git checkout --orphan main-clean`), commit the cleaned tree as a fresh root, and replace `main` with it. This rewrites `main` and needs a force-push to origin. Do not do this unless the user explicitly asks for a history-free `main`; it is destructive to the shared branch.

Decision needed before executing: non-destructive archival (recommended) versus an orphan/fresh `main`. Acceptance: `origin/v1_deprecated` shows the pre-cleanup v1 and `origin/main` shows the cleaned version, with nothing lost. Files: none (git only).

--------------------------------------------------------------------------------

## Task 1 — Order candidates by quality so the zoom never opens on junk

Problem: `candidates()` returns contours in OpenCV's find order (roughly spatial), and the tuner selects index 0, so the zoom often opens "heavily zoomed into a random spot on the ground" instead of the gate. `process_frame()` sorts its output, but the tuner reads the unsorted `candidates` list from `process_frame_debug`.

Change: sort the list returned by `candidates()` deterministically, best first — passed candidates before rejected, then by `confidence` descending, then `area_frac` descending. The tuner's default `selected_idx = 0` then lands on the strongest candidate. Keep `process_frame()`'s existing closest-first (area) sort for production output, or have it read from the same ordered list and re-sort by area; either way production behavior is unchanged.

Acceptance: on a frame containing a gate, candidate `0/N` in the zoom is the gate, not a floor blob. Files: `gate_detection_core.py` (sort inside `candidates()`); no tuner change required.

--------------------------------------------------------------------------------

## Task 2 — Split and shrink morphology so geometry and the inner hole survive

Problem: one `morph_kernel` is used for both the close and the open, and large values round the square outline into a blob (ruining corner extraction) and seal the gate's inner window (so `inner_corners` is always "none").

Change: replace the single `morph_kernel` with two parameters, `close_kernel` and `open_kernel` (each forced odd, elliptical), defaulting small (start at 3–5). Apply `MORPH_CLOSE` with `close_kernel`, then `MORPH_OPEN` with `open_kernel`. Keep `morph_kernel` as a deprecated alias that, if present in a loaded config, sets both new kernels, so existing `gate_config.json` files still load. In the tuner, replace the one `morph_kernel` trackbar with two (`close_kernel`, `open_kernel`). Document in the code and README that a large close kernel fills the gate's inner window and defeats inner-corner detection, so the close kernel should be only just large enough to bridge the LED gaps in the colored ring.

Acceptance: at tuned settings the cleaned mask keeps a recognizably square outline and a visible inner hole, and inner corners are found on a clearly visible gate. Files: `gate_detection_core.py` (params + `clean_mask`), `tuner.py` (trackbars + panel label), `gate_config.json` (schema/defaults).

--------------------------------------------------------------------------------

## Task 3 — Score gates on hollowness, not raw size

Problem: confidence is `0.5*solidity + 0.5*size_score`, which rejects distant real gates for being small and cannot separate a hollow gate from a solid disc of the same size. Depends on Task 2 having restored the inner hole.

Change: introduce a hollowness signal and let it drive confidence instead of raw size. Compute `has_inner` (a valid 4-point inner hole exists for this contour) and `fill_ratio = contour_area / bbox_area` (a thin ring is well below 1; a solid blob is high). Define `hollowness_score = 1.0` if a valid inner quad exists, else `clamp(1 - fill_ratio, 0, 1)`. Redefine confidence as a weighted sum with configurable weights and a deliberately small size weight: `confidence = w_solidity*solidity + w_hollow*hollowness_score + w_size*size_score`, recommended defaults `w_solidity = 0.4`, `w_hollow = 0.4`, `w_size = 0.2`. Keep `min_area_frac` as the hard size gate so size still filters out specks but no longer dominates the score. Surface the hollowness term in the tuner's zoom stats next to solidity and confidence.

Acceptance: a sharp distant gate above `min_area_frac` is accepted, and a solid round floor reflection of similar size is either rejected or scores clearly lower than a real gate. Files: `gate_detection_core.py` (metrics + confidence + config weights), `gate_config.json`, `tuner.py` (zoom readout).

--------------------------------------------------------------------------------

## Task 4 — Corner refinement (cast the true gate corners, not a box)

Problem: outer corners come from `approxPolyDP` run on a morph-distorted blob (at `approx_epsilon_frac` around 0.065), and when the approximation does not return exactly four points the code falls back to the `minAreaRect` box, which discards the gate's true perspective. The drawn corners drift off the real gate corners, and on angled gates the box fallback is visibly wrong.

Change: with Task 2 providing a cleaner contour, add a side-line refinement. Take the four `approxPolyDP` corners as seeds; split the contour's points into the four side-segments that run between consecutive seeds; fit a line to each side with `cv2.fitLine`; intersect adjacent side-lines to get four sub-pixel corners; order them `[TL, TR, BR, BL]` as before. Fall back to `minAreaRect` only when a side has too few points to fit, and mark any box-fallback candidate as lower confidence so it is visibly distinguished. Apply the same refinement to the inner-hole corners.

Acceptance: across several frames the outer corners sit on the visible gate corners rather than inset or rounded, including on angled gates where the box fallback used to be clearly off. Files: `gate_detection_core.py` (corner extraction).

--------------------------------------------------------------------------------

## Task 5 — Per-constraint views so each color slider tunes in isolation

Problem: the color mask is hue AND saturation AND value, but the tuner shows only the finished band mask and the raw saturation channel; the hue channel is never shown, so hue-band tuning is blind. This is the "we apply the other filters over them" concern.

Change: add an inspection mode to the tuner, toggled by a key (`i`), that swaps the mosaic to the isolated contributions of each constraint: (1) hue visualized as color, mapping the H channel to a full-saturation, full-value BGR image so the gate's hue is directly readable; (2) a hue-only mask (`inRange` on H alone, ignoring S and V); (3) a saturation-only mask (`s >= sat_min`); (4) a value-only mask (`val_min <= v <= val_max`); (5) `mask_raw` (the bands OR'd, before morphology) shown beside (6) `mask_clean` so the morphology's effect is visible; plus the usual candidates and result panels. Emit these isolated masks from `process_frame_debug` so the layout code stays thin. Keep the normal 8-panel mosaic as the default and have `i` toggle between the normal and inspection layouts.

Acceptance: in inspection mode, dragging only `sat_min` visibly moves the saturation-only mask while the hue-only and value-only masks hold still, and the hue-as-color view makes the gate's hue band obvious. Files: `gate_detection_core.py` (`process_frame_debug` emits the isolated masks and the colorized hue image), `tuner.py` (inspection layout + `i` toggle), README (document the mode).

--------------------------------------------------------------------------------

## Task 6 — Undo the last control change (Ctrl+Z) and show the prior value

Problem: there is no way to revert a slider or toggle you just changed, and the previous value is lost the moment you move it.

Change: route every parameter write through a single choke point (the existing `apply_controls`) that, before applying the new values, pushes a snapshot of `detector.to_dict()` plus the full set of trackbar positions onto a bounded undo stack (cap depth around 50). Bind Ctrl+Z (which `cv2.waitKeyEx` returns as `26`) to pop the stack and restore both the detector parameters and every trackbar position via `cv2.setTrackbarPos`, then flag the frame dirty so the views refresh. Show the most recent change and its reversibility on the status bar, for example `last: sat_min 37->42  [Ctrl+Z undo]`, which also satisfies the request to keep the pre-change state visible on screen.

Honest OpenCV limitation to design around: highgui does not expose which window currently has OS focus, so "Ctrl+Z while focused on a specific panel" cannot be scoped reliably to one control panel. Specify a single global undo stack shared across both control panels (recommended, simplest, and predictable). If per-panel undo is still wanted, approximate it by recording which panel's trackbar changed last and undoing within that panel, and document that it is best-effort.

Acceptance: move any slider, press Ctrl+Z, and both the value and the slider position return to the previous state; repeat to walk back several changes. Files: `tuner.py`.

--------------------------------------------------------------------------------

## Task 7 — Arrow-key slider control (keyboard, no mouse)

Problem: sliders can currently only be dragged with the mouse, which is imprecise for a +/-1 nudge.

Change: because OpenCV trackbars have no native focus or selection concept, the tuner maintains its own selection model — an ordered list of the trackbars in each control panel plus a cursor index, and one "active" panel at a time. Up and Down move the cursor to the previous/next slider in the active panel; Left and Right decrement/increment the selected slider by 1 via `getTrackbarPos`/`setTrackbarPos`; the selected slider's name is drawn in a small highlighted header strip in that panel so the user can see what is selected. Because OpenCV cannot report which of `tune: color` and `tune: shape` is active, use Tab to switch the active panel and indicate which one is active. Reassign keys so nothing is overloaded: arrows drive the slider cursor, `a`/`d` and `A`/`D` drive frame stepping, and `[`/`]` drive candidate selection — so the current Left/Right = frame-step binding moves entirely onto `a`/`d`.

Acceptance: with no mouse, Tab picks a panel, Up/Down select a slider, and Left/Right nudge it by 1 with the masks and result updating live. Files: `tuner.py`, README (updated key map).

--------------------------------------------------------------------------------

## Task 8 — Gate-zoom overhaul (locator box, context margin, readable corners)

Problem: the zoom crops straight to the candidate's bounding box with no surrounding context, so it opens "heavily zoomed into a random spot" with no sense of where in the frame it is looking; the corner overlay is hard to read; and the default candidate is junk (fixed by Task 1).

Change, four parts. (a) Locator panel: add a full-frame overview — the whole `proc` image dimmed to roughly 40% brightness — with the selected candidate's crop region drawn on it as a bright, visually distinct rectangle (thick, high-contrast, e.g. cyan) so it is immediately obvious what the zoom is showing; make it the first tile of the `gate zoom` window. (b) Context margin: before upscaling, pad the crop by a margin (about 25% of the bbox on each side, or a fixed minimum number of pixels for tiny candidates) and clamp to the frame bounds, so a small candidate shows real surrounding context instead of a meaningless nearest-neighbor smear. (c) Corner readability: draw the ordered outer corners numbered 0–3 in TL,TR,BR,BL order with the reprojected quad edges, draw the inner corners in a distinct color, and label whether the corners came from Task 4's line-fit refinement or the `minAreaRect` box fallback. (d) Default selection: rely on Task 1 so candidate 0 is the best; keep `[`/`]` to cycle.

Acceptance: opening the zoom lands on a real gate, the locator box clearly marks the crop region on the full frame, tiny candidates still show context, and the corners are legible and correctly ordered. Files: `tuner.py` (`build_zoom` plus a new `build_locator` helper); depends on Task 1 and Task 4.

--------------------------------------------------------------------------------

## Task 9 — README: full four-window breakdown, per-slider docs, markdown hygiene

Problem: the README does not clearly explain the `gate zoom` window, does not document every slider, and still references the removed overlay.

Change, three parts. (a) Add a "Windows" section documenting all four windows element by element: for `pipeline`, each of the eight (and the inspection-mode) panels — what it shows, why it exists, how it is computed; for `gate zoom`, each element — the locator box, the proc/mask/saturation crops, the numbered corners, and the stats readout with the meaning of each stat (band, area_frac, aspect, solidity, hollowness, confidence, result); for `tune: color` and `tune: shape`, a one-row-per-slider table giving what the slider does, why it is adjustable, and which direction has which effect. (b) Remove every mention of `gate_overlay.py` and the overlay workflow, since it is deleted. (c) Markdown hygiene: write each paragraph as a single continuous line with no mid-paragraph hard line breaks; break lines only between paragraphs, list items, and table rows. This paragraph-per-line rule applies to all docs from now on, this spec included.

Acceptance: a new reader can understand all four windows and every slider from the README alone, with no overlay references and no hard-wrapped paragraphs. Files: `README.md`.

--------------------------------------------------------------------------------

## Task 10 — Purge the overlay from the repo

Problem: the overlay is removed from disk, but traces remain in the requirements, the docs, and the core's docstring.

Change: confirm `gate_overlay.py` is deleted (done); remove `mss` from `requirements.txt` if nothing else uses it (only the overlay did); fix the `gate_detection_core.py` module docstring line that still reads "Shared by the overlay and the tuner" to name only the tuner; and grep the repo for `overlay`, `mss`, `gate_overlay`, and `screen capture`, removing any remaining stale references in the docs (README, and the v1 sections of this spec). Leave this roadmap's mentions as the historical record of the removal.

Acceptance: a case-insensitive grep for `overlay` over the code and docs returns only this roadmap's lines, and `requirements.txt` lists no unused dependency. Files: `requirements.txt`, `gate_detection_core.py` (docstring), `README.md`, this spec.
