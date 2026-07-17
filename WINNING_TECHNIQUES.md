# Winning-Strategy Gate Detection

This detector implements techniques from the **2019 AIRR (Artificial Intelligence Robotic Racing) Competition Winner** - a winning team from Delft University that achieved 9.2 m/s (33 km/h) autonomous racing speeds.

## Key Techniques

### Stage 0: Preprocessing (Semantic Segmentation)
**Why:** Motion blur and lighting variations are major challenges at high speeds.

**Technique:** Bilateral filtering
- Smooths noise while preserving sharp edges
- Critical for motion blur robustness
- Trained data included artificial motion blur (5-15 pixel blur at random angles)

**Tuning:**
- `Bilateral Sigma`: Higher = more smoothing (default 10)
- Too high = loses gate details
- Too low = keeps too much noise

---

### Stage 1: YOLO Detection
**Why:** Need initial localization of gate regions before detailed analysis.

**Technique:** YOLOv8 Nano lightweight model
- Processes full image at ~50 Hz
- Confidence threshold set LOW (0.35) to catch distant gates
- Later stages filter false positives

**Tuning:**
- `Confidence Threshold`: Lower detects more gates (including distant ones)
  - **0.3-0.4**: Catches all gates, more false positives
  - **0.5-0.6**: Balanced
  - **0.7+**: Only high-confidence gates

---

### Stage 2: Semantic Segmentation (Mask Extraction)
**Why:** Bounding boxes alone are insufficient. Gates are complex shapes with multiple corners. Need pixel-level segmentation.

**Technique:** Multi-method fusion with weighted voting
- Each method detects different aspects of the gate
- Methods are fused via weighted combination
- Confidence calculated from method agreement

#### Method 1: Edge Detection (Canny)
**Detects:** Gate frame/boundary edges
- Robust to color variations
- Works even when gate content is obscured
- Sensitive to motion blur unless properly augmented

**Tuning:**
- `Canny Low`: Lower threshold (default 40) - detects fainter edges
- `Canny High`: Upper threshold (default 120) - rejects noise
- Rule of thumb: High/Low ratio ~3:1
- If getting too many edges: **increase both thresholds**
- If missing edges: **decrease both thresholds**
- `Edge Weight`: How much edge detection contributes to final mask
  - 1.0 = equal weight with other methods
  - \>1.0 = emphasize edges more
  - <1.0 = rely on other methods more

#### Method 2: Color Detection (HSV)
**Detects:** Bright and colored gate markers (logos, paint, reflective tape)
- Robust to shape/perspective changes
- Best for gates with distinct coloring
- Ignores grayscale backgrounds

**Tuning:**
- `Hue Min`: Which color range to detect (0-180 in OpenCV HSV)
  - 0-30: Red/Orange
  - 30-60: Yellow
  - 60-90: Green
  - 90-120: Cyan/Blue
  - 120-150: Blue/Magenta
  - For mixed colors or white/bright regions: start at 0
- `Saturation Min`: Minimum color saturation (0-255)
  - Lower = detect pale colors too
  - Higher = only vivid colors
  - **Tip:** Start around 80 to catch gate markers
- `Value Min`: Minimum brightness (0-255)
  - Lower = detect dark regions too
  - Higher = only bright regions
  - **Tip:** Start around 120 to catch lit gates
- `Color Weight`: How much color detection contributes
  - Increase if gates are brightly colored
  - Decrease if too many false color detections

**Pro Tip:** The winning team found that **logos and checkerboard patterns** on gates were critical. Even with low contrast (80% transparency), gates could be detected down to 30% of image size.

#### Method 3: Corner Detection (Harris)
**Detects:** Corner regions and interest points
- Robust to rotation and scale
- Works on any textured surface
- Helps find gate corners precisely

**Tuning:**
- `Corner Sensitivity`: Threshold for corner strength (0.001-0.05)
  - Lower = detect weaker corners (more detections)
  - Higher = only strong corners (fewer false positives)
  - **Default 0.008 is good compromise**
- `Corner Weight`: How much corners contribute

---

### Stage 3: Active Vision Corner Finding
**Why:** Don't scan entire gate region. Winning paper found that **active vision** (intelligent sampling) is much faster and more accurate.

**Technique:** "Snake-gate" algorithm
- Finds vertical and horizontal density histograms of gate mask
- Starts from region of highest density
- Walks outward in 4 directions until boundary found
- Only samples ~1.64% of pixels instead of 100%
- Executes in 0.23ms

**Performance:**
- Traditional contour finding: Slow, sensitive to discontinuous masks
- Active vision: Fast, robust to gaps and overlaps

**Tuning:**
- `Active Vision Enabled`: Toggle on/off
- `Corner Search Radius`: How far to search from center (default 15 pixels)
  - Increase if corners are far from gate center
  - Decrease for efficiency if gates are clean

---

### Stage 4: Gate-Prior Validation (Sanity Check)
**Why:** Reject false positives by validating against expected gate shape. Winning paper used this to handle **overlapping gates** and **moving gates**.

**Technique:** Aspect ratio validation
- Expected gate dimensions come from YOLO bounding box
- Compare detected gate aspect ratio to expected
- Reject if error exceeds tolerance
- Can auto-correct minor corner errors using expected shape

**How it works:**
1. Calculate expected gate dimensions (from YOLO box)
2. Measure detected gate dimensions
3. Compute aspect ratio error
4. Reject if error > tolerance

**Tuning:**
- `Gate Prior Validation Enabled`: Toggle on/off
- `Gate Prior Tolerance`: Maximum allowed aspect ratio error
  - 0.25 (25%) = strict validation, paper used this
  - 0.35 (35%) = more permissive, our default
  - \>0.5 = very loose, almost never rejects

**When to adjust:**
- **Gates being rejected?** Increase tolerance or disable gate-prior
- **Too many false positives?** Decrease tolerance or enable gate-prior
- **Gates at extreme angles?** Increase tolerance (perspective distortion)

---

## Tuning Strategy for Your Video

### Step 1: Get Any Detection
1. Lower YOLO confidence to 0.2-0.3 (aggressive)
2. Enable all detection methods
3. Run and see if gates appear at all

### Step 2: Reduce False Positives
Once gates are detected:
1. Enable gate-prior validation
2. Increase YOLO confidence gradually (0.3 → 0.5)
3. Reduce color detection saturation minimum if gates have muted colors
4. Increase edge detection thresholds if getting too many edges

### Step 3: Perfect Each Method
Fine-tune individual methods:
1. **Enable ONLY edge detection** - tune Canny thresholds
2. **Enable ONLY color detection** - tune HSV ranges
3. **Enable ONLY corner detection** - tune sensitivity
4. **Enable all three** - fine-tune method weights

### Step 4: Optimize for Your Video
Once all methods work:
1. Adjust `confidence_threshold` to skip obvious false positives
2. Fine-tune method weights to emphasize most reliable methods
3. Adjust `gate_prior_tolerance` based on gate orientations

---

## Performance Metrics

The winning 2019 AIRR team achieved:
- **GateNet IoU (Intersection over Union):** 87% on championship race footage
- **Detection speed:** 75.9 Hz (13.18ms) for gate segmentation
- **Active vision:** 0.23ms (40%+ faster than alternatives)
- **Average racing speed:** 6.75 m/s (24 km/h)
- **Max racing speed:** 9.19 m/s (33 km/h)

Key insight: They tripled previous autonomous racing speeds by:
1. Efficient perception (active vision sampling)
2. Multi-method fusion (handles environmental variation)
3. Robust validation (gate-prior sanity checks)
4. Smart preprocessing (trained on motion blur)

---

## References

De Wagter, C., Paredes-Vallés, F., Sheth, N., & de Croon, G. C. H. E. (2022).
"The sensing, state-estimation, and control behind the winning entry to the 2019 artificial intelligence robotic racing competition."
Field Robotics, 2, 1263–1290.

Paper describes:
- GateNet semantic segmentation network
- Snake-gate active vision algorithm
- Gate-prior corner validation
- Moving Horizon Estimator (MHE) for state fusion
- Risk-aware control strategy
