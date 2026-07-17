# Notes from the 2019 AIRR winning entry

Reference for design decisions. The paper is De Wagter et al. (2022), "The
sensing, state-estimation, and control behind the winning entry to the 2019
Artificial Intelligence Robotic Racing competition," Field Robotics 2, 1263-1290.
Implementation specifics live in `BUILD_SPEC_TUNER.md`; this file only records what
we adopted and what we rejected, and why.

## What we adopted for Part 1

- Appearance-based segmentation of the gate, not bounding-box object detection.
  The gate is found from how it looks (its color/shape), not from a generic object
  detector. In our footage the gates are a saturated color against a desaturated
  background, so a saturation-gated HSV threshold is the segmentation.
- A shape prior to reject false positives. The paper validates gate geometry; we
  validate aspect ratio, solidity, and minimum size on the color blob.
- Robustness to motion blur via light preprocessing (bilateral filter) plus a
  segmentation cue (color) that survives blur better than fine edges.

## What we rejected, and why

- A stock COCO object detector (YOLOv8n). It is trained on people/cars/etc., has
  never seen a gate, and is the only component that needs heavy compute. Removed.
- Equal-weight fusion of Canny edges + HSV color + Harris corners. Averaging three
  weak cues produced mush and fired on the arena's lights and structural lines.
  Replaced by a single strong cue (color) plus geometric validation.
- The paper's "snake-gate" active-vision corner finder. It solves corner-finding
  inside a segmentation mask; `cv2.minAreaRect` on the color blob does the same job
  more simply and more robustly for our footage. Removed.

## What is deferred to Part 2

- A trained gate detector (the paper's GateNet is a segmentation network trained on
  labeled gate images). This is the robust long-term answer, especially for the
  more complex Virtual Qualifier 2 environment, but it requires labeled gate data
  we do not have yet. Part 1 reaches reliable detection with classical color CV
  first; the trained model is a later upgrade, not a prerequisite.

## Key numbers from the paper (context)

Average racing speed 6.75 m/s, max 9.19 m/s; gate segmentation ran at ~76 Hz. They
tripled previous autonomous racing speeds primarily through efficient perception,
robust validation, and preprocessing that anticipated motion blur.
