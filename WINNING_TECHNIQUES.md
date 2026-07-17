# Notes from the 2019 AIRR winning entry

Reference for design decisions. Paper: De Wagter et al. (2022), "The sensing,
state-estimation, and control behind the winning entry to the 2019 Artificial
Intelligence Robotic Racing competition," Field Robotics 2, 1263-1290.
Implementation specifics live in `BUILD_SPEC_TUNER.md`; this file records what we
took from the paper, what we changed, and how their approach maps onto our project.

## Their full perception pipeline

Four sequential stages turn an image into a drone position:

1. GateNet — a U-Net neural network outputs a binary mask of all visible gates
   (regardless of distance). Trained on 2336 hand-annotated images.
2. Snake-gate — active-vision search that reads the mask's row/column histograms
   and samples outward to find the gate's inner and outer corners (up to 8 points).
3. Gate-prior — projects the expected gate into the image and validates the found
   corners against it; rejects or corrects corners whose sides/angles are off by
   more than 25%. This exists to patch snake-gate's failure cases.
4. Perspective-n-Point (PnP) — feeds the 8 corners plus the known gate size into
   OpenCV's PnP solver to recover the drone's 3D position and orientation.

Corners are the linchpin: they are how a 2D mask becomes a 3D pose. Everything
downstream (state estimation, control) depends on them.

## What Part 1 does, mapped to their pipeline

- Mask (their stage 1): we use an HSV saturation-gated color threshold instead of a
  trained network. Our gates are a saturated color against a desaturated arena, so
  color isolates them. The mask stage is deliberately swappable — a trained network
  replaces it in Part 2 without touching the rest.
- Corners (their stage 2): we keep corner finding, but via contour + approxPolyDP
  instead of snake-gate. approxPolyDP returns the true four-corner quadrilateral
  (so an angled gate gives its real trapezoid, not a bounding box), and the colored
  ring's contour hierarchy gives inner and outer corners the same way snake-gate
  does. We rejected snake-gate specifically because it requires a continuous mask
  and no gate overlap — conditions that broke on most AIRR tracks.
- Validation (their stage 3): we validate corners by aspect ratio, solidity, and
  minimum size. Same purpose as gate-prior, simpler mechanism.
- PnP (their stage 4): deferred to Part 2. Part 1 produces and orders the corners
  so PnP can consume them later.

## What we rejected outright

- A stock COCO object detector (YOLOv8n): never trained on gates, and the only
  component that needs heavy compute. Removed.
- Equal-weight fusion of Canny + HSV + Harris: averaging three weak cues produced
  mush and fired on lights and structural lines. Replaced by one strong cue plus
  geometric validation.

## Training data and simulation (how they crossed the reality gap)

Directly relevant to our roadmap, because our competition also provides a simulator.

- Data was a mix of real and synthetic. Their 2336 training images came from 8
  environments; one of those was a simulator, the rest were real venues. Masks were
  hand-annotated.
- They did not rely on photorealism. Robustness came from data augmentation: random
  affine transforms, HSV color-space shifts, and artificial motion blur (averaging
  filter of random size 5-15 px and random orientation).
- The competition organizers provided a hardware-in-the-loop (HiL) simulator that
  ran the drone dynamics and produced synthetic images. They used it both as a
  training-data source and to validate state estimation against ground truth that
  real flights never provide ("no position ground truth is available for the
  competition flights").
- They used simulation for controlled ablations: varying gate scale, logos,
  checkerboards, and transparency to learn what the network keys on. Finding: the
  network exploits contrast from logos/checkerboards, not specific shapes; it
  detects gates down to ~30% of image size and degrades past ~80% logo transparency.
- Each race got its own GateNet, fine-tuned on ~50 hand-labeled hard frames from
  that venue's practice session (incremental training).

How this maps to us: the AI Grand Prix simulator is simultaneously our competition
environment, our labeled-data generator (the sim knows exact gate/drone positions,
so labels are automatic — no manual annotation), and our ground-truth validator.
For VQ1/VQ2 the sim IS the target, so there is no reality gap yet; it appears only
at the physical stages, where their augmentation recipe (HSV, affine, motion blur)
becomes the tool for crossing it. Part 2 is: generate labeled gate images from the
simulator, train a segmentation network to replace the color mask, and add PnP to
turn corners into drone position.

## Key numbers from the paper (context)

Average racing speed 6.75 m/s, max 9.19 m/s. GateNet forward pass 13.18 ms
(~76 Hz); full perception pipeline ~17 ms at ~54 Hz. They tripled previous
autonomous racing speeds through efficient perception, robust corner validation,
and preprocessing that anticipated motion blur.
