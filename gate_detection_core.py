"""
Gate detection core (Part 1): pure classical-CV, color-first gate detection.

No neural network, no UI, no capture, no threads. Takes a BGR frame, returns
gate detections. Shared by the overlay and the tuner. See BUILD_SPEC_TUNER.md.
"""

import json
import time

import cv2
import numpy as np


DEFAULTS = {
    "proc_width": 640,

    "preprocess_enabled": True,
    "bilateral_sigma": 10,

    "band_a_enabled": True,
    "band_a_h_lo": 120,
    "band_a_h_hi": 160,

    "band_b_enabled": False,
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
    "find_inner_corners": True,

    "confidence_threshold": 0.30,
}


class GateDetector:
    def __init__(self, config=None):
        for key, value in DEFAULTS.items():
            setattr(self, key, value)
        if config:
            self.update(**config)

    # ---- config helpers -------------------------------------------------

    def to_dict(self):
        return {key: getattr(self, key) for key in DEFAULTS}

    def update(self, **params):
        for key, value in params.items():
            if key in DEFAULTS:
                setattr(self, key, value)

    def load_config(self, path):
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        self.update(**data)

    def save_config(self, path):
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_config_file(cls, path):
        detector = cls()
        detector.load_config(path)
        return detector

    # ---- geometry helper ------------------------------------------------

    def _resize_to_proc(self, bgr):
        h, w = bgr.shape[:2]
        proc_w = int(self.proc_width)
        proc_h = max(1, int(round(h * proc_w / w)))
        resized = cv2.resize(bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        scale = w / float(proc_w)  # multiply proc-space coords by this -> original
        return resized, scale

    def _kernel(self):
        k = int(self.morph_kernel)
        if k % 2 == 0:
            k += 1
        k = max(1, k)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ---- primitives -----------------------------------------------------

    def preprocess(self, bgr):
        if not self.preprocess_enabled:
            return bgr
        return cv2.bilateralFilter(bgr, 9, self.bilateral_sigma, self.bilateral_sigma)

    def color_mask(self, bgr):
        """Return (raw OR'd mask, band_a mask, band_b mask). No morphology here."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]
        empty = np.zeros((h, w), dtype=np.uint8)

        if self.band_a_enabled:
            mask_a = cv2.inRange(
                hsv,
                (int(self.band_a_h_lo), int(self.sat_min), int(self.val_min)),
                (int(self.band_a_h_hi), 255, int(self.val_max)),
            )
        else:
            mask_a = empty.copy()

        if self.band_b_enabled:
            mask_b = cv2.inRange(
                hsv,
                (int(self.band_b_h_lo), int(self.sat_min), int(self.val_min)),
                (int(self.band_b_h_hi), 255, int(self.val_max)),
            )
        else:
            mask_b = empty.copy()

        mask = cv2.bitwise_or(mask_a, mask_b)
        return mask, mask_a, mask_b

    def clean_mask(self, mask_raw):
        kernel = self._kernel()
        cleaned = cv2.morphologyEx(mask_raw, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def order_corners(self, pts):
        """Return the 4 points ordered [top-left, top-right, bottom-right, bottom-left]."""
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
        by_y = pts[np.argsort(pts[:, 1])]
        top = by_y[:2]
        bottom = by_y[2:]
        top = top[np.argsort(top[:, 0])]        # TL, TR
        bottom = bottom[np.argsort(bottom[:, 0])]  # BL, BR
        tl, tr = top[0], top[1]
        bl, br = bottom[0], bottom[1]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _band_of(self, contour, shape, mask_a, mask_b):
        fill = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(fill, [contour], -1, 255, -1)
        overlap_a = cv2.countNonZero(cv2.bitwise_and(fill, mask_a))
        overlap_b = cv2.countNonZero(cv2.bitwise_and(fill, mask_b))
        if overlap_a == 0 and overlap_b == 0:
            return "A"
        hi = max(overlap_a, overlap_b)
        lo = min(overlap_a, overlap_b)
        if overlap_a > 0 and overlap_b > 0 and (hi - lo) <= 0.20 * hi:
            return "A+B"
        return "A" if overlap_a >= overlap_b else "B"

    def _inner_corners(self, contours, hierarchy, parent_idx):
        """Largest child hole of parent_idx -> ordered 4 corners, or None."""
        if hierarchy is None:
            return None
        best = None
        best_area = 0.0
        for i, node in enumerate(hierarchy[0]):
            if node[3] == parent_idx:
                area = cv2.contourArea(contours[i])
                if area > best_area:
                    best_area = area
                    best = contours[i]
        if best is None:
            return None
        peri = cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, self.approx_epsilon_frac * peri, True)
        if len(approx) == 4:
            return self.order_corners(approx.reshape(4, 2))
        return None

    def candidates(self, mask_clean, mask_a, mask_b):
        """Steps 6-10 in proc space. Returns per-contour dicts (passed and rejected)."""
        proc_h, proc_w = mask_clean.shape[:2]
        frame_area = float(proc_w * proc_h)
        contours, hierarchy = cv2.findContours(
            mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        results = []
        for idx, contour in enumerate(contours):
            # Only level-0 contours (outer boundaries) are gate candidates.
            if hierarchy is not None and hierarchy[0][idx][3] != -1:
                continue
            if len(contour) < 3:
                continue
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            area_frac = area / frame_area

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), _angle = rect
            aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
            hull = cv2.convexHull(contour)
            solidity = area / max(cv2.contourArea(hull), 1e-6)

            # Outer corners: true quad via approxPolyDP, else minAreaRect box.
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.approx_epsilon_frac * peri, True)
            if len(approx) == 4:
                outer_corners = self.order_corners(approx.reshape(4, 2))
            else:
                outer_corners = self.order_corners(cv2.boxPoints(rect))

            inner_corners = None
            if self.find_inner_corners:
                inner_corners = self._inner_corners(contours, hierarchy, idx)

            size_score = min(1.0, area_frac / 0.02)
            confidence = 0.5 * solidity + 0.5 * size_score

            reject_reason = ""
            if area_frac < self.min_area_frac:
                reject_reason = "area"
            elif not (self.min_aspect <= aspect <= self.max_aspect):
                reject_reason = "aspect"
            elif solidity < self.min_solidity:
                reject_reason = "solidity"
            elif confidence < self.confidence_threshold:
                reject_reason = "confidence"
            passed = reject_reason == ""

            band = self._band_of(contour, mask_clean.shape, mask_a, mask_b)

            x, y, bw, bh = cv2.boundingRect(contour)
            results.append({
                "outer_corners": outer_corners,
                "inner_corners": inner_corners,
                "center": (float(cx), float(cy)),
                "bbox": (x, y, x + bw, y + bh),
                "aspect": float(aspect),
                "solidity": float(solidity),
                "area_frac": float(area_frac),
                "confidence": float(confidence),
                "band": band,
                "passed": passed,
                "reject_reason": reject_reason,
            })
        return results

    # ---- entry points ---------------------------------------------------

    def process_frame(self, bgr):
        resized, scale = self._resize_to_proc(bgr)
        pre = self.preprocess(resized)
        mask_raw, mask_a, mask_b = self.color_mask(pre)
        mask_clean = self.clean_mask(mask_raw)
        cands = self.candidates(mask_clean, mask_a, mask_b)

        gates = []
        for cand in cands:
            if not cand["passed"]:
                continue
            outer = (cand["outer_corners"] * scale).astype(np.float32)
            inner = cand["inner_corners"]
            if inner is not None:
                inner = (inner * scale).astype(np.float32)
            cx, cy = cand["center"]
            cx, cy = cx * scale, cy * scale
            bx1, by1, bx2, by2 = [int(round(v * scale)) for v in cand["bbox"]]
            gates.append({
                "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                "width": bx2 - bx1, "height": by2 - by1,
                "center_x": int(round(cx)), "center_y": int(round(cy)),
                "corners": outer,
                "inner_corners": inner,
                "confidence": cand["confidence"],
                "band": cand["band"],
                "aspect": cand["aspect"],
                "solidity": cand["solidity"],
                "area_frac": cand["area_frac"],
            })
        gates.sort(key=lambda g: g["area_frac"], reverse=True)
        return gates

    def process_frame_debug(self, bgr):
        t0 = time.time()
        resized, scale = self._resize_to_proc(bgr)
        pre = self.preprocess(resized)

        t_color = time.time()
        mask_raw, mask_a, mask_b = self.color_mask(pre)
        mask_clean = self.clean_mask(mask_raw)
        color_ms = (time.time() - t_color) * 1000.0

        hsv = cv2.cvtColor(pre, cv2.COLOR_BGR2HSV)
        hsv_h, hsv_s, hsv_v = cv2.split(hsv)

        t_cont = time.time()
        cands = self.candidates(mask_clean, mask_a, mask_b)
        contours_ms = (time.time() - t_cont) * 1000.0

        total_ms = (time.time() - t0) * 1000.0
        return {
            "proc": resized,
            "preprocessed": pre,
            "hsv_h": hsv_h,
            "hsv_s": hsv_s,
            "hsv_v": hsv_v,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "mask_raw": mask_raw,
            "mask_clean": mask_clean,
            "candidates": cands,
            "scale": scale,
            "timings": {
                "color_ms": color_ms,
                "contours_ms": contours_ms,
                "total_ms": total_ms,
            },
        }
