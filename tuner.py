"""
Visual tuner for the Part 1 color-first gate detector (BUILD_SPEC_TUNER.md).

OpenCV only: four resizable windows ("pipeline", "gate zoom", and two control
panels), integer trackbars, a dirty-flag blocking loop, an 8-panel mosaic and a per-candidate
zoom. No tkinter, no matplotlib, no threads. Importing this module opens no
window: all GUI/window/trackbar/event-loop code lives under main().

The mosaic and zoom are built by standalone functions (build_mosaic /
build_zoom) that never call cv2.imshow, so they can be exercised headless.
"""

import datetime
import glob
import json
import os
import sys

import cv2
import numpy as np

from gate_detection_core import GateDetector


FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CONFIG_PATH = "gate_config.json"

# waitKeyEx extended codes for the arrow keys (Windows/Qt highgui).
KEY_LEFT = 2424832
KEY_RIGHT = 2555904

# Color-pick: hue half-width set around a clicked pixel's hue.
HUE_TOL = 10


class FrameSource:
    """Frames from a video file OR a folder of images (sorted by name). Lets the
    tuner run on captured PNG frames, not just an .mp4."""
    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, path):
        self.is_dir = os.path.isdir(path)
        if self.is_dir:
            names = sorted(os.listdir(path))
            self.paths = [os.path.join(path, n) for n in names
                          if n.lower().endswith(self.IMG_EXT)]
            self.count = len(self.paths)
            print("Loaded %d image frames from %s" % (self.count, path))
        else:
            self.cap = cv2.VideoCapture(path)
            ok = self.cap.isOpened()
            self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
            if ok and self.count <= 0:
                self.count = 1

    def read(self, idx):
        if self.is_dir:
            if 0 <= idx < self.count:
                img = cv2.imread(self.paths[idx])
                return img is not None, img
            return False, None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        return self.cap.read()

    def release(self):
        if not self.is_dir:
            self.cap.release()


def conf_color(conf):
    """Confidence -> BGR color: green >0.8, yellow >0.5, else red."""
    if conf > 0.8:
        return GREEN
    if conf > 0.5:
        return YELLOW
    return RED


def _to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _label(img, text):
    """Draw a top-left label: white text over a 1px black outline."""
    cv2.putText(img, text, (5, 15), FONT, 0.4, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, text, (5, 15), FONT, 0.4, WHITE, 1, cv2.LINE_AA)


def _make_cell(panel, text, cell_w, cell_h):
    cell = cv2.resize(_to_bgr(panel), (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    _label(cell, text)
    return cell


def _poly(corners):
    return np.round(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)


def build_mosaic(result, detector, selected_idx, frame_idx=0, frame_count=0):
    """Build the 2x4 pipeline mosaic + status bar. Never calls cv2.imshow."""
    proc = result["proc"]
    ph, pw = proc.shape[:2]
    cell_w = 320
    cell_h = max(1, int(round(cell_w * ph / float(pw))))
    cands = result["candidates"]

    # Panel 7: candidates (minAreaRect polygon not stored on the candidate
    # dict; outer_corners is the per-candidate quad the core exposes).
    p7 = proc.copy()
    for cand in cands:
        color = GREEN if cand["passed"] else RED
        cv2.polylines(p7, [_poly(cand["outer_corners"])], True, color, 1, cv2.LINE_AA)
        if cand["reject_reason"]:
            cx, cy = cand["center"]
            cv2.putText(p7, cand["reject_reason"], (int(cx) - 15, int(cy)),
                        FONT, 0.35, color, 1, cv2.LINE_AA)

    # Panel 8: result (passed gates colored by confidence; selected in cyan).
    p8 = proc.copy()
    for cand in cands:
        if not cand["passed"]:
            continue
        cv2.polylines(p8, [_poly(cand["outer_corners"])], True,
                      conf_color(cand["confidence"]), 2, cv2.LINE_AA)
    if 0 <= selected_idx < len(cands):
        sel = cands[selected_idx]
        cv2.polylines(p8, [_poly(sel["outer_corners"])], True, CYAN, 3, cv2.LINE_AA)
        scx, scy = sel["center"]
        cv2.circle(p8, (int(round(scx)), int(round(scy))), 4, CYAN, -1)

    panels = [
        (proc, "1 original"),
        (result["preprocessed"], "2 preprocess s=%d" % detector.bilateral_sigma),
        (result["hsv_s"], "3 saturation"),
        (result["mask_a"], "4 band A H%d-%d" % (detector.band_a_h_lo, detector.band_a_h_hi)),
        (result["mask_b"], "5 band B H%d-%d" % (detector.band_b_h_lo, detector.band_b_h_hi)),
        (result["mask_clean"], "6 mask (morph k=%d)" % detector.morph_kernel),
        (p7, "7 candidates"),
        (p8, "8 result"),
    ]
    cells = [_make_cell(p, t, cell_w, cell_h) for p, t in panels]

    grid_w = cell_w * 4
    grid_h = cell_h * 2
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = i // 4, i % 4
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

    # 2px black borders separating cells (drawn in place to keep width == 1280).
    canvas[cell_h - 1:cell_h + 1, :] = 0
    for c in (1, 2, 3):
        x = c * cell_w
        canvas[:, x - 1:x + 1] = 0

    n_pass = sum(1 for cc in cands if cc["passed"])
    n_cand = len(cands)
    t = result["timings"]
    bar = np.zeros((24, grid_w, 3), dtype=np.uint8)
    status = ("frame %d/%d  passed %d/%d  color %.1fms total %.1fms  "
              "[SPACE play][a/d step][ [ ] select ][w save cfg][s snapshot]"
              % (frame_idx, frame_count, n_pass, n_cand,
                 t["color_ms"], t["total_ms"]))
    cv2.putText(bar, status, (6, 17), FONT, 0.4, WHITE, 1, cv2.LINE_AA)

    return cv2.vconcat([canvas, bar])


def build_zoom(result, detector, selected_idx):
    """Build the selected-candidate zoom breakdown. Never calls cv2.imshow."""
    cands = result["candidates"]
    n = len(cands)
    if n == 0 or selected_idx < 0 or selected_idx >= n:
        img = np.zeros((240, 480, 3), dtype=np.uint8)
        cv2.putText(img, "no candidates", (20, 120), FONT, 0.7, WHITE, 1, cv2.LINE_AA)
        return img

    cand = cands[selected_idx]
    proc = result["proc"]
    mask_clean = result["mask_clean"]
    hsv_s = result["hsv_s"]
    ph, pw = proc.shape[:2]

    x1, y1, x2, y2 = cand["bbox"]
    x1 = max(0, min(int(x1), pw - 1))
    y1 = max(0, min(int(y1), ph - 1))
    x2 = max(x1 + 1, min(int(x2), pw))
    y2 = max(y1 + 1, min(int(y2), ph))

    crop_proc = proc[y1:y2, x1:x2].copy()
    crop_mask = mask_clean[y1:y2, x1:x2]
    crop_s = hsv_s[y1:y2, x1:x2]

    cw = max(1, x2 - x1)
    factor = max(1, int(np.ceil(240.0 / cw)))

    up_proc = cv2.resize(crop_proc, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_NEAREST)
    up_mask = _to_bgr(cv2.resize(crop_mask, None, fx=factor, fy=factor,
                                 interpolation=cv2.INTER_NEAREST))
    up_s = _to_bgr(cv2.resize(crop_s, None, fx=factor, fy=factor,
                              interpolation=cv2.INTER_NEAREST))

    for i, (px, py) in enumerate(cand["outer_corners"]):
        pt = (int(round((px - x1) * factor)), int(round((py - y1) * factor)))
        cv2.circle(up_proc, pt, 4, GREEN, -1)
        cv2.putText(up_proc, str(i), (pt[0] + 4, pt[1] - 4), FONT, 0.4, GREEN, 1, cv2.LINE_AA)
    if cand["inner_corners"] is not None:
        for i, (px, py) in enumerate(cand["inner_corners"]):
            pt = (int(round((px - x1) * factor)), int(round((py - y1) * factor)))
            cv2.circle(up_proc, pt, 4, CYAN, -1)
            cv2.putText(up_proc, str(i), (pt[0] + 4, pt[1] - 4), FONT, 0.4, CYAN, 1, cv2.LINE_AA)

    _label(up_proc, "proc")
    _label(up_mask, "mask")
    _label(up_s, "sat")
    left = cv2.vconcat([up_proc, up_mask, up_s])

    inner_txt = "4" if cand["inner_corners"] is not None else "none"
    res_txt = "PASS" if cand["passed"] else "REJECT (%s)" % cand["reject_reason"]
    lines = [
        ("candidate %d/%d" % (selected_idx, n), WHITE),
        ("band        %s" % cand["band"], WHITE),
        ("corners     outer 4 / inner %s" % inner_txt, WHITE),
        ("area_frac   %.4f   (min %s)" % (cand["area_frac"], detector.min_area_frac), WHITE),
        ("aspect      %.2f   (%s..%s)" % (cand["aspect"], detector.min_aspect, detector.max_aspect), WHITE),
        ("solidity    %.2f   (min %s)" % (cand["solidity"], detector.min_solidity), WHITE),
        ("confidence  %.2f   (thr %s)" % (cand["confidence"], detector.confidence_threshold),
         conf_color(cand["confidence"])),
        ("result      %s" % res_txt, GREEN if cand["passed"] else RED),
    ]

    line_step = 26
    right_w = 340
    target_h = max(left.shape[0], 20 + len(lines) * line_step)
    if left.shape[0] < target_h:
        pad = np.zeros((target_h - left.shape[0], left.shape[1], 3), dtype=np.uint8)
        left = cv2.vconcat([left, pad])
    right = np.zeros((target_h, right_w, 3), dtype=np.uint8)

    y = 28
    for text, color in lines:
        cv2.putText(right, text, (10, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        y += line_step

    return cv2.hconcat([left, right])


def build_color_view(result):
    """'Dog vision': the color filter's output -- gate-colored pixels kept in full
    color, everything else dimmed to gray. Click it to sample a hue into band A."""
    proc = result["proc"]
    mask = result["mask_raw"]
    gray = cv2.cvtColor(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = (gray * 0.35).astype(np.uint8)
    out[mask > 0] = proc[mask > 0]
    _label(out, "color filter: what the drone 'sees' (click a gate pixel to pick color)")
    return out


def _clamp(v, lo, hi):
    return max(lo, min(int(round(v)), hi))


def main():
    args = sys.argv[1:]
    if args:
        media_path = args[0]
    else:
        mp4s = sorted(glob.glob("*.mp4"))
        if not mp4s:
            print("Usage: tuner.py <video-file-or-image-folder> [config.json]")
            return
        media_path = mp4s[0]
    config_path = args[1] if len(args) > 1 else CONFIG_PATH

    detector = GateDetector.from_config_file(config_path)
    print("Editing config -> %s" % os.path.abspath(config_path))
    print("Press 'w' to SAVE your changes to that file.")

    source = FrameSource(media_path)
    count = source.count
    if count <= 0:
        print("Could not open: %s" % media_path)
        return

    state = {
        "frame_idx": 0,
        "frame_changed": True,
        "dirty": True,
        "playing": False,
        "selected_idx": 0,
        "frame": None,
        "result": None,
        "ready": False,
    }

    # OpenCV trackbars cannot scroll, and one window with 17 of them grows taller
    # than the screen. Split them across two short, resizable, movable panels.
    WIN_COLOR = "tune: color"
    WIN_SHAPE = "tune: shape"
    for name in ("pipeline", "gate zoom", WIN_COLOR, WIN_SHAPE):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    placeholder = np.zeros((24, 360, 3), dtype=np.uint8)
    cv2.imshow(WIN_COLOR, placeholder)
    cv2.imshow(WIN_SHAPE, placeholder)
    cv2.resizeWindow("pipeline", 1280, 528)
    cv2.resizeWindow(WIN_COLOR, 380, 430)
    cv2.resizeWindow(WIN_SHAPE, 380, 400)
    cv2.moveWindow("pipeline", 0, 0)
    cv2.moveWindow("gate zoom", 0, 600)
    cv2.moveWindow(WIN_COLOR, 1290, 0)
    cv2.moveWindow(WIN_SHAPE, 1290, 470)

    cv2.namedWindow("color view", cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("color view", 650, 600)

    tb_win = {}

    def on_frame(pos):
        state["frame_idx"] = pos
        state["frame_changed"] = True

    def add_tb(name, window, init, maxv, cb):
        cv2.createTrackbar(name, window, init, maxv, cb)
        tb_win[name] = window

    def apply_controls(_=0):
        if not state["ready"]:
            return
        g = lambda name: cv2.getTrackbarPos(name, tb_win[name])
        mk = max(1, g("morph_kernel"))
        if mk % 2 == 0:
            mk += 1
        detector.update(
            proc_width=g("proc_width") + 320,
            bilateral_sigma=max(1, g("bilateral_sigma")),
            band_a_h_lo=g("band_a_h_lo"),
            band_a_h_hi=g("band_a_h_hi"),
            band_b_h_lo=g("band_b_h_lo"),
            band_b_h_hi=g("band_b_h_hi"),
            sat_min=g("sat_min"),
            val_min=g("val_min"),
            val_max=g("val_max"),
            morph_kernel=mk,
            min_area_frac=max(1, g("min_area x10000")) / 10000.0,
            min_solidity=g("min_solidity x100") / 100.0,
            min_aspect=max(10, g("min_aspect x100")) / 100.0,
            max_aspect=max(10, g("max_aspect x100")) / 100.0,
            approx_epsilon_frac=max(5, g("approx_eps x1000")) / 1000.0,
            confidence_threshold=g("gate_conf x100") / 100.0,
        )
        state["dirty"] = True

    # Color / preprocessing panel
    add_tb("proc_width", WIN_COLOR, _clamp(detector.proc_width - 320, 0, 960), 960, apply_controls)
    add_tb("bilateral_sigma", WIN_COLOR, _clamp(detector.bilateral_sigma, 1, 30), 30, apply_controls)
    add_tb("band_a_h_lo", WIN_COLOR, _clamp(detector.band_a_h_lo, 0, 180), 180, apply_controls)
    add_tb("band_a_h_hi", WIN_COLOR, _clamp(detector.band_a_h_hi, 0, 180), 180, apply_controls)
    add_tb("band_b_h_lo", WIN_COLOR, _clamp(detector.band_b_h_lo, 0, 180), 180, apply_controls)
    add_tb("band_b_h_hi", WIN_COLOR, _clamp(detector.band_b_h_hi, 0, 180), 180, apply_controls)
    add_tb("sat_min", WIN_COLOR, _clamp(detector.sat_min, 0, 255), 255, apply_controls)
    add_tb("val_min", WIN_COLOR, _clamp(detector.val_min, 0, 255), 255, apply_controls)
    add_tb("val_max", WIN_COLOR, _clamp(detector.val_max, 0, 255), 255, apply_controls)

    # Frame navigation + shape/validation panel
    add_tb("frame", WIN_SHAPE, 0, max(1, count - 1), on_frame)
    add_tb("morph_kernel", WIN_SHAPE, _clamp(detector.morph_kernel, 1, 21), 21, apply_controls)
    add_tb("min_area x10000", WIN_SHAPE, _clamp(detector.min_area_frac * 10000, 1, 500), 500, apply_controls)
    add_tb("min_solidity x100", WIN_SHAPE, _clamp(detector.min_solidity * 100, 0, 100), 100, apply_controls)
    add_tb("min_aspect x100", WIN_SHAPE, _clamp(detector.min_aspect * 100, 10, 300), 300, apply_controls)
    add_tb("max_aspect x100", WIN_SHAPE, _clamp(detector.max_aspect * 100, 10, 500), 500, apply_controls)
    add_tb("approx_eps x1000", WIN_SHAPE, _clamp(detector.approx_epsilon_frac * 1000, 5, 100), 100, apply_controls)
    add_tb("gate_conf x100", WIN_SHAPE, _clamp(detector.confidence_threshold * 100, 0, 100), 100, apply_controls)

    state["ready"] = True
    apply_controls()

    def on_color_pick(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state["result"] is None:
            return
        r = state["result"]
        hh, ss, vv = r["hsv_h"], r["hsv_s"], r["hsv_v"]
        ph, pw = hh.shape[:2]
        if not (0 <= x < pw and 0 <= y < ph):
            return
        h, s, v = int(hh[y, x]), int(ss[y, x]), int(vv[y, x])
        lo, hi = max(0, h - HUE_TOL), min(180, h + HUE_TOL)
        smin, vmin = max(0, s - 50), max(0, v - 60)
        print("picked HSV=(%d,%d,%d) -> band A H[%d-%d] sat_min %d val_min %d"
              % (h, s, v, lo, hi, smin, vmin))
        cv2.setTrackbarPos("band_a_h_lo", tb_win["band_a_h_lo"], lo)
        cv2.setTrackbarPos("band_a_h_hi", tb_win["band_a_h_hi"], hi)
        cv2.setTrackbarPos("sat_min", tb_win["sat_min"], smin)
        cv2.setTrackbarPos("val_min", tb_win["val_min"], vmin)
        # Update the detector directly too, so the pick can't be lost if
        # setTrackbarPos doesn't fire the slider callback on this OpenCV build.
        detector.update(band_a_enabled=True, band_a_h_lo=lo, band_a_h_hi=hi,
                        sat_min=smin, val_min=vmin)
        state["dirty"] = True

    cv2.setMouseCallback("color view", on_color_pick)

    def step(delta):
        ni = max(0, min(state["frame_idx"] + delta, count - 1))
        state["frame_idx"] = ni
        cv2.setTrackbarPos("frame", tb_win["frame"], ni)
        state["frame_changed"] = True

    def select(delta):
        if state["result"] is None:
            return
        n = len(state["result"]["candidates"])
        if n == 0:
            return
        state["selected_idx"] = max(0, min(state["selected_idx"] + delta, n - 1))
        state["dirty"] = True

    def snapshot():
        if state["result"] is None:
            return
        os.makedirs("snapshots", exist_ok=True)
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mosaic = build_mosaic(state["result"], detector, state["selected_idx"],
                              state["frame_idx"], count)
        png_path = os.path.join("snapshots", "snap_%s_mosaic.png" % ts)
        json_path = os.path.join("snapshots", "snap_%s_params.json" % ts)
        cv2.imwrite(png_path, mosaic)
        params = detector.to_dict()
        params["frame"] = state["frame_idx"]
        with open(json_path, "w") as fh:
            json.dump(params, fh, indent=2)
        print("Saved snapshot %s" % png_path)

    while True:
        if state["frame_changed"]:
            ok, frame = source.read(state["frame_idx"])
            if ok:
                state["frame"] = frame
            state["frame_changed"] = False
            state["dirty"] = True

        if state["dirty"] and state["frame"] is not None:
            result = detector.process_frame_debug(state["frame"])
            state["result"] = result
            n = len(result["candidates"])
            state["selected_idx"] = 0 if n == 0 else max(0, min(state["selected_idx"], n - 1))
            cv2.imshow("pipeline", build_mosaic(result, detector, state["selected_idx"],
                                                state["frame_idx"], count))
            cv2.imshow("gate zoom", build_zoom(result, detector, state["selected_idx"]))
            cv2.imshow("color view", build_color_view(result))
            state["dirty"] = False

        key = cv2.waitKeyEx(30)
        quit_requested = False
        if key != -1:
            k = key & 0xFF
            if key == KEY_RIGHT or k == ord("d"):
                step(1)
            elif key == KEY_LEFT or k == ord("a"):
                step(-1)
            elif k == ord("D"):
                step(30)
            elif k == ord("A"):
                step(-30)
            elif k == 32:
                state["playing"] = not state["playing"]
            elif k == ord("["):
                select(-1)
            elif k == ord("]"):
                select(1)
            elif k == ord("p"):
                detector.update(preprocess_enabled=not detector.preprocess_enabled)
                state["dirty"] = True
            elif k == ord("1"):
                detector.update(band_a_enabled=not detector.band_a_enabled)
                state["dirty"] = True
            elif k == ord("2"):
                detector.update(band_b_enabled=not detector.band_b_enabled)
                state["dirty"] = True
            elif k == ord("w"):
                detector.save_config(config_path)
                print("SAVED config -> %s" % os.path.abspath(config_path))
            elif k == ord("s"):
                snapshot()
            elif k == ord("q") or key == 27:
                quit_requested = True
        if quit_requested:
            break

        if state["playing"]:
            if state["frame_idx"] < count - 1:
                step(1)
            else:
                state["playing"] = False

    source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
