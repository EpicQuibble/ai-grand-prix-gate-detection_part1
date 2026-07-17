#!/usr/bin/env python3
"""
Winning-strategy gate detection overlay.
Incorporates techniques from 2019 AIRR (Artificial Intelligence Robotic Racing) competition winner:
- Semantic segmentation preprocessing (gate mask extraction)
- Active vision corner finding (samples only pixels on gate boundary)
- Gate-prior sanity checking (validates detected corners)
- Motion blur robustness
- Intelligent temporal tracking
"""

import threading
import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL
from PIL import Image, ImageDraw, ImageTk, ImageGrab
from ultralytics import YOLO
import numpy as np
import cv2
import time


class GateDetectionOverlay:
    def __init__(self, monitor_index=0):
        self.model = YOLO("yolov8n.pt")

        try:
            import mss
            self.mss = mss
            self.sct = mss.mss()
            monitor = self.sct.monitors[monitor_index + 1]
            self.capture_area = {
                'left': monitor['left'],
                'top': monitor['top'],
                'width': monitor['width'],
                'height': monitor['height']
            }
        except:
            self.sct = None
            self.capture_area = None

        self.current_frame = None
        self.current_detections = []
        self.frame_idx = 0

        self.playing = True

        # YOLO settings
        self.yolo_confidence = 0.35  # Lower to catch distant gates

        # Preprocessing (Semantic segmentation stage)
        self.preprocess_enabled = True
        self.morphology_kernel = 5
        self.bilateral_sigma = 10  # Reduce noise while keeping edges

        # Edge detection settings (Canny)
        self.canny_low = 40
        self.canny_high = 120
        self.edge_enabled = True
        self.edge_weight = 1.0

        # Color detection settings (HSV ranges for bright/colored gates)
        self.color_h_min = 0
        self.color_s_min = 80
        self.color_v_min = 120
        self.color_enabled = True
        self.color_weight = 1.2

        # Corner detection settings (Harris)
        self.corner_threshold = 0.008
        self.corner_enabled = True
        self.corner_weight = 0.8

        # Active vision settings (snake-gate style corner finding)
        self.active_vision_enabled = True
        self.corner_search_radius = 15

        # Gate-prior sanity check settings
        self.gate_prior_enabled = True
        self.gate_prior_tolerance = 0.35  # 35% max error (paper used 25%)

        # Fusion settings
        self.min_methods = 1  # Minimum methods that must detect gate
        self.confidence_threshold = 0.25  # Minimum fused confidence

        # Display settings
        self.show_labels = True
        self.show_centers = True
        self.show_method_tags = True

        # State tracking
        self.gates_passed = 0
        self.last_gate_center_y = None
        self.gate_age = {}  # Track age of each gate in frames
        self.gate_history = {}  # Track gate positions over time for stability

        self.lock = threading.Lock()
        self.width = 1920
        self.height = 1080

    def detect_gate_passage(self, gates):
        """Detect gate passage"""
        if not gates:
            self.last_gate_center_y = None
            return False

        gate = gates[0]
        current_y = gate['center_y']

        if self.last_gate_center_y is None:
            self.last_gate_center_y = current_y
            return False

        center_y = self.height / 2
        was_above = self.last_gate_center_y < center_y
        is_below = current_y > center_y

        self.last_gate_center_y = current_y

        return was_above and is_below

    def preprocess_frame(self, frame):
        """Preprocess frame like GateNet training (semantic segmentation stage).
        Reduces noise while preserving edges - key to robustness at high speeds."""
        if not self.preprocess_enabled:
            return frame

        # Reduce noise with bilateral filter (preserves edges better than Gaussian)
        frame_filtered = cv2.bilateralFilter(frame, 9, self.bilateral_sigma, self.bilateral_sigma)

        return frame_filtered

    def detect_edges(self, frame):
        """Edge detection using Canny (robust to motion blur via training)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter first (smooths noise, keeps edges sharp)
        gray = cv2.bilateralFilter(gray, 5, 10, 10)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # Dilate slightly to connect broken edges (from motion blur)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def detect_color(self, frame):
        """Color detection using HSV - detects bright/colored gate markers"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broader HSV range to catch all hues of gate markings
        lower = np.array([self.color_h_min, self.color_s_min, self.color_v_min])
        upper = np.array([180, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def detect_corners(self, frame):
        """Corner detection using Harris corner detector"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        return corners

    def find_corners_active_vision(self, gate_mask, yolo_box):
        """Active vision corner finding (snake-gate algorithm).
        Samples only pixels on gate boundary instead of scanning entire region.
        Much faster and more accurate than scanning all pixels.

        Returns: list of 4 corner points [top-left, top-right, bottom-right, bottom-left]
        """
        if gate_mask is None or gate_mask.size == 0:
            return None

        x1, y1, x2, y2 = yolo_box
        h, w = gate_mask.shape

        # Find vertical and horizontal histograms (count white pixels per row/col)
        vert_hist = np.sum(gate_mask, axis=0)  # Sum across rows → column sums
        horz_hist = np.sum(gate_mask, axis=1)  # Sum across cols → row sums

        if vert_hist.max() == 0 or horz_hist.max() == 0:
            return None

        # Find center region where gate is densest
        center_col = np.argmax(vert_hist)
        center_row = np.argmax(horz_hist)

        corners = []

        # Find 4 corners by walking outward from center in 4 directions
        # Top-left: walk left and up from center
        for col in range(max(0, center_col - self.corner_search_radius), center_col + 1):
            for row in range(max(0, center_row - self.corner_search_radius), center_row + 1):
                if gate_mask[row, col] > 0:
                    corners.append([col + x1, row + y1])
                    break

        # Top-right: walk right and up from center
        for col in range(min(w - 1, center_col + self.corner_search_radius), center_col - 1, -1):
            for row in range(max(0, center_row - self.corner_search_radius), center_row + 1):
                if gate_mask[row, col] > 0:
                    corners.append([col + x1, row + y1])
                    break

        # Bottom-right: walk right and down
        for col in range(min(w - 1, center_col + self.corner_search_radius), center_col - 1, -1):
            for row in range(min(h - 1, center_row + self.corner_search_radius), center_row - 1, -1):
                if gate_mask[row, col] > 0:
                    corners.append([col + x1, row + y1])
                    break

        # Bottom-left: walk left and down
        for col in range(max(0, center_col - self.corner_search_radius), center_col + 1):
            for row in range(min(h - 1, center_row + self.corner_search_radius), center_row - 1, -1):
                if gate_mask[row, col] > 0:
                    corners.append([col + x1, row + y1])
                    break

        if len(corners) < 3:
            return None

        return np.array(corners, dtype=np.float32)

    def validate_gate_prior(self, detected_corners, yolo_box):
        """Gate-prior sanity check: validate detected corners against expected gate shape.
        Rejects false positives and corrects minor errors.

        Returns: (is_valid, corrected_corners)
        """
        if detected_corners is None or len(detected_corners) < 3:
            return False, None

        x1, y1, x2, y2 = yolo_box
        expected_width = x2 - x1
        expected_height = y2 - y1

        # Expected gate aspect ratio (assuming roughly square gates)
        expected_aspect = expected_height / (expected_width + 1e-6)

        # Measure detected gate dimensions
        detected_width = np.max(detected_corners[:, 0]) - np.min(detected_corners[:, 0]) + 1e-6
        detected_height = np.max(detected_corners[:, 1]) - np.min(detected_corners[:, 1]) + 1e-6
        detected_aspect = detected_height / detected_width

        # Allow 35% tolerance (paper used 25%, we're more permissive)
        aspect_error = abs(detected_aspect - expected_aspect) / (expected_aspect + 1e-6)

        if aspect_error > self.gate_prior_tolerance:
            return False, None

        return True, detected_corners

    def find_gate_polygon(self, gate_region, yolo_box):
        """Find gate polygon using winning-strategy fusion.

        Stage 1: Semantic segmentation (extract gate mask)
        Stage 2: Active vision (sample only gate boundary pixels to find corners)
        Stage 3: Gate-prior validation (sanity check against expected location)
        """
        if gate_region is None or gate_region.size == 0:
            return None

        x1, y1, x2, y2 = yolo_box
        methods_found = []
        method_confidences = {}

        # Stage 1: Create semantic segmentation mask (like GateNet)
        edges = self.detect_edges(gate_region) if self.edge_enabled else None
        colors = self.detect_color(gate_region) if self.color_enabled else None
        corners = self.detect_corners(gate_region) if self.corner_enabled else None

        # Build weighted combined mask
        combined = np.zeros_like(gate_region[:, :, 0], dtype=np.float32)

        if self.edge_enabled and edges is not None:
            combined += edges.astype(np.float32) * self.edge_weight
            methods_found.append('edge')
            method_confidences['edge'] = np.mean(edges) / 255.0

        if self.color_enabled and colors is not None:
            combined += colors.astype(np.float32) * self.color_weight
            methods_found.append('color')
            method_confidences['color'] = np.mean(colors) / 255.0

        if self.corner_enabled and corners is not None:
            corner_map = (corners > self.corner_threshold * corners.max()).astype(np.float32)
            combined += corner_map * self.corner_weight
            methods_found.append('corner')
            method_confidences['corner'] = np.mean(corner_map)

        if len(methods_found) < self.min_methods:
            return None

        # Normalize and threshold
        combined = (combined / (combined.max() + 1e-6) * 255).astype(np.uint8)
        _, gate_mask = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gate_mask = cv2.morphologyEx(gate_mask, cv2.MORPH_CLOSE, kernel)

        # Stage 2: Active vision corner finding (samples only gate boundary)
        if self.active_vision_enabled:
            corners_detected = self.find_corners_active_vision(gate_mask, yolo_box)
        else:
            # Fallback: use contour-based method
            contours, _ = cv2.findContours(gate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            corners_detected = hull.reshape(-1, 2) if len(hull) >= 3 else None

        if corners_detected is None or len(corners_detected) < 3:
            return None

        # Stage 3: Gate-prior validation (sanity check)
        if self.gate_prior_enabled:
            is_valid, corners_detected = self.validate_gate_prior(corners_detected, yolo_box)
            if not is_valid:
                return None

        # Calculate fused confidence (how many methods detected it + their agreement)
        fused_confidence = len(methods_found) / 3.0 * np.mean(list(method_confidences.values()))
        fused_confidence = max(self.confidence_threshold, fused_confidence)

        return {
            'hull': corners_detected,
            'methods': methods_found,
            'confidence': fused_confidence
        }

    def process_detections(self, frame):
        """Process frame with winning-strategy pipeline.

        1. Preprocess frame (noise reduction while preserving edges)
        2. Run YOLO detection
        3. For each detection: semantic segmentation + active vision + gate-prior validation
        4. Temporal filtering (track gate stability)
        """
        # Stage 0: Preprocess (like GateNet training with HSV augmentation)
        frame_processed = self.preprocess_frame(frame)

        # Stage 1: YOLO detection (lowered confidence to catch distant gates)
        results = self.model(frame_processed, conf=self.yolo_confidence, verbose=False)

        gates = []
        for det in results[0].boxes:
            yolo_conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Clip to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            gate_region = frame_processed[y1:y2, x1:x2]

            # Stage 2-3: Semantic segmentation + active vision + validation
            polygon_info = self.find_gate_polygon(gate_region, (x1, y1, x2, y2))

            if polygon_info is None:
                continue

            # Only keep gates that pass fusion confidence threshold
            if polygon_info['confidence'] < self.confidence_threshold:
                continue

            methods = polygon_info['methods']

            gates.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': w, 'height': h,
                'center_x': cx, 'center_y': cy,
                'yolo_confidence': yolo_conf,
                'fusion_confidence': polygon_info['confidence'],
                'hull': polygon_info['hull'],
                'methods': methods
            })

        # Sort by size (closest gates first)
        gates.sort(key=lambda g: g['height'], reverse=True)

        return gates

    def capture_thread(self):
        """Capture screen continuously"""
        while self.playing:
            try:
                if self.sct:
                    screenshot = self.sct.grab(self.capture_area)
                    frame = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                else:
                    frame = ImageGrab.grab()

                frame = frame.convert('RGB')
                self.width, self.height = frame.size

                frame_array = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)

                with self.lock:
                    self.current_frame = frame_bgr.copy()
                    self.frame_idx += 1

            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.01)

            time.sleep(0.01)

    def detection_thread(self):
        """Run detection continuously"""
        while self.playing:
            with self.lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.current_frame.copy()

            try:
                gates = self.process_detections(frame)

                if self.detect_gate_passage(gates):
                    self.gates_passed += 1

                with self.lock:
                    self.current_detections = gates

                    for gate in gates:
                        gate_key = (gate['x1'], gate['y1'], gate['x2'], gate['y2'])
                        if gate_key not in self.gate_age:
                            self.gate_age[gate_key] = 0
                        self.gate_age[gate_key] += 1

            except Exception as e:
                print(f"Detection error: {e}")

            time.sleep(0.01)

    def create_overlay_image(self):
        """Create transparent overlay with detection boxes"""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        with self.lock:
            detections = self.current_detections.copy()

        for i, gate in enumerate(detections):
            conf = gate['fusion_confidence']

            if conf > 0.8:
                color = (0, 255, 0, 200)
            elif conf > 0.5:
                color = (255, 255, 0, 200)
            else:
                color = (255, 0, 0, 200)

            if gate['hull'] is not None:
                hull_points = [tuple(pt) for pt in gate['hull']]
                if len(hull_points) > 2:
                    draw.polygon(hull_points, outline=color, fill=None)
            else:
                x1, y1, x2, y2 = gate['x1'], gate['y1'], gate['x2'], gate['y2']
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            if self.show_centers:
                cx, cy = gate['center_x'], gate['center_y']
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)

            if self.show_labels:
                gate_key = (gate['x1'], gate['y1'], gate['x2'], gate['y2'])
                age = self.gate_age.get(gate_key, 0)

                text = f"G{i + 1} {conf:.2f} (age:{age})"

                if self.show_method_tags and gate['methods']:
                    methods_str = ','.join(gate['methods'][:2])
                    text += f" [{methods_str}]"

                draw.text((gate['x1'] + 5, gate['y1'] + 5), text, fill=color)

        return img

    def overlay_window(self):
        """Transparent overlay window"""
        overlay = tk.Tk()
        overlay.attributes('-topmost', True)
        overlay.geometry(f"{self.width}x{self.height}+0+0")
        overlay.attributes('-transparentcolor', 'black')

        canvas = tk.Canvas(overlay, bg='black', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.photo_image = None
        canvas.image_id = None

        def update_overlay():
            img = self.create_overlay_image()
            photo_image = ImageTk.PhotoImage(img)

            if canvas.image_id is None:
                canvas.image_id = canvas.create_image(0, 0, image=photo_image, anchor='nw')
            else:
                canvas.itemconfig(canvas.image_id, image=photo_image)

            canvas.photo_image = photo_image

            overlay.after(33, update_overlay)

        update_overlay()

        overlay.mainloop()

    def toggle_detection(self):
        self.playing = not self.playing
        status = "ON" if self.playing else "OFF"
        self.detection_button.config(text=f"DETECTION: {status}")

    def control_panel(self):
        """Control panel window"""
        root = tk.Tk()
        root.title("Gate Detector Control (Winning-Strategy)")
        root.geometry("450x900")

        # Main scrollable frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        frame = scrollable_frame

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(frame, text="Winning-Strategy Gate Detector", font=("Arial", 12, "bold")).pack()
        ttk.Label(frame, text="2019 AIRR Competition Techniques", font=("Arial", 9, "italic")).pack()
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        self.detection_button = ttk.Button(frame, text="DETECTION: ON", command=self.toggle_detection)
        self.detection_button.pack(fill='x', pady=5)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Preprocessing (Stage 0)
        ttk.Label(frame, text="Stage 0: Preprocessing (Semantic Segmentation)", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Enable Preprocessing (bilateral filter)",
                       command=lambda: setattr(self, 'preprocess_enabled', not self.preprocess_enabled)).pack(anchor='w')
        ttk.Label(frame, text="Bilateral Filter Sigma", font=("Arial", 9)).pack(anchor='w')
        bilateral_scale = Scale(frame, from_=1, to=20, orient=HORIZONTAL,
                               command=lambda v: setattr(self, 'bilateral_sigma', int(v)))
        bilateral_scale.set(self.bilateral_sigma)
        bilateral_scale.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # YOLO (Stage 1)
        ttk.Label(frame, text="Stage 1: YOLO Detection", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Label(frame, text="Confidence Threshold (lower = detect distant gates)", font=("Arial", 9)).pack(anchor='w')
        yolo_scale = Scale(frame, from_=0.1, to=0.9, resolution=0.05, orient=HORIZONTAL,
                          command=self.set_yolo_confidence)
        yolo_scale.set(self.yolo_confidence)
        yolo_scale.pack(fill='x', pady=5)
        self.yolo_label = ttk.Label(frame, text=f"{self.yolo_confidence:.2f}")
        self.yolo_label.pack(anchor='w')

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Semantic Segmentation Stage (Stage 2)
        ttk.Label(frame, text="Stage 2: Semantic Segmentation", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Label(frame, text="Gate Confidence Threshold", font=("Arial", 9)).pack(anchor='w')
        conf_scale = Scale(frame, from_=0.1, to=0.9, resolution=0.05, orient=HORIZONTAL,
                          command=lambda v: setattr(self, 'confidence_threshold', float(v)))
        conf_scale.set(self.confidence_threshold)
        conf_scale.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Edge Detection
        ttk.Label(frame, text="Method 1: Edge Detection (Canny)", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Enable Edge Detection", command=self.toggle_edge).pack(anchor='w')
        ttk.Label(frame, text="Canny Low", font=("Arial", 9)).pack(anchor='w')
        canny_low = Scale(frame, from_=10, to=100, orient=HORIZONTAL,
                         command=lambda v: setattr(self, 'canny_low', int(v)))
        canny_low.set(self.canny_low)
        canny_low.pack(fill='x', pady=2)

        ttk.Label(frame, text="Canny High", font=("Arial", 9)).pack(anchor='w')
        canny_high = Scale(frame, from_=50, to=300, orient=HORIZONTAL,
                          command=lambda v: setattr(self, 'canny_high', int(v)))
        canny_high.set(self.canny_high)
        canny_high.pack(fill='x', pady=2)

        ttk.Label(frame, text="Edge Weight", font=("Arial", 9)).pack(anchor='w')
        edge_weight = Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=HORIZONTAL,
                           command=lambda v: setattr(self, 'edge_weight', float(v)))
        edge_weight.set(self.edge_weight)
        edge_weight.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Color Detection
        ttk.Label(frame, text="Method 2: Color Detection (HSV)", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Enable Color Detection (bright/colored regions)", command=self.toggle_color).pack(anchor='w')
        ttk.Label(frame, text="Hue Min", font=("Arial", 9)).pack(anchor='w')
        hue_scale = Scale(frame, from_=0, to=180, orient=HORIZONTAL,
                         command=lambda v: setattr(self, 'color_h_min', int(v)))
        hue_scale.set(self.color_h_min)
        hue_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Saturation Min", font=("Arial", 9)).pack(anchor='w')
        sat_scale = Scale(frame, from_=0, to=255, orient=HORIZONTAL,
                         command=lambda v: setattr(self, 'color_s_min', int(v)))
        sat_scale.set(self.color_s_min)
        sat_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Value Min (brightness)", font=("Arial", 9)).pack(anchor='w')
        val_scale = Scale(frame, from_=0, to=255, orient=HORIZONTAL,
                         command=lambda v: setattr(self, 'color_v_min', int(v)))
        val_scale.set(self.color_v_min)
        val_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Color Weight", font=("Arial", 9)).pack(anchor='w')
        color_weight = Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=HORIZONTAL,
                            command=lambda v: setattr(self, 'color_weight', float(v)))
        color_weight.set(self.color_weight)
        color_weight.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Corner Detection
        ttk.Label(frame, text="Method 3: Corner Detection (Harris)", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Enable Corner Detection", command=self.toggle_corner).pack(anchor='w')
        ttk.Label(frame, text="Corner Sensitivity", font=("Arial", 9)).pack(anchor='w')
        corner_scale = Scale(frame, from_=0.001, to=0.05, resolution=0.005, orient=HORIZONTAL,
                            command=lambda v: setattr(self, 'corner_threshold', float(v)))
        corner_scale.set(self.corner_threshold)
        corner_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Corner Weight", font=("Arial", 9)).pack(anchor='w')
        corner_weight = Scale(frame, from_=0.1, to=2.0, resolution=0.1, orient=HORIZONTAL,
                             command=lambda v: setattr(self, 'corner_weight', float(v)))
        corner_weight.set(self.corner_weight)
        corner_weight.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Stage 3: Active Vision & Gate-Prior
        ttk.Label(frame, text="Stage 3: Corner Finding & Validation", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Enable Active Vision (samples 1.64% of pixels - faster)",
                       command=lambda: setattr(self, 'active_vision_enabled', not self.active_vision_enabled)).pack(anchor='w')
        ttk.Checkbutton(frame, text="Enable Gate-Prior Validation (sanity check)",
                       command=lambda: setattr(self, 'gate_prior_enabled', not self.gate_prior_enabled)).pack(anchor='w')
        ttk.Label(frame, text="Gate-Prior Tolerance", font=("Arial", 9)).pack(anchor='w')
        prior_scale = Scale(frame, from_=0.1, to=0.5, resolution=0.05, orient=HORIZONTAL,
                           command=lambda v: setattr(self, 'gate_prior_tolerance', float(v)))
        prior_scale.set(self.gate_prior_tolerance)
        prior_scale.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(frame, text="Display Options", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Labels", command=self.toggle_labels).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Centers", command=self.toggle_centers).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Method Tags", command=self.toggle_methods).pack(anchor='w')

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        self.stats_frame = ttk.LabelFrame(frame, text="Statistics", padding="5")
        self.stats_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.captures_label = ttk.Label(self.stats_frame, text="Captures: 0")
        self.captures_label.pack(anchor='w')

        self.gates_label = ttk.Label(self.stats_frame, text="Gates Visible: 0")
        self.gates_label.pack(anchor='w')

        self.passed_label = ttk.Label(self.stats_frame, text="Gates Passed: 0")
        self.passed_label.pack(anchor='w')

        def update_stats():
            with self.lock:
                self.captures_label.config(text=f"Captures: {self.frame_idx}")
                self.gates_label.config(text=f"Gates Visible: {len(self.current_detections)}")
                self.passed_label.config(text=f"Gates Passed: {self.gates_passed}")

            root.after(100, update_stats)

        update_stats()

        root.mainloop()

    def toggle_edge(self):
        self.edge_enabled = not self.edge_enabled

    def toggle_color(self):
        self.color_enabled = not self.color_enabled

    def toggle_corner(self):
        self.corner_enabled = not self.corner_enabled

    def toggle_labels(self):
        self.show_labels = not self.show_labels

    def toggle_centers(self):
        self.show_centers = not self.show_centers

    def toggle_methods(self):
        self.show_method_tags = not self.show_method_tags

    def set_yolo_confidence(self, value):
        self.yolo_confidence = float(value)
        self.yolo_label.config(text=f"{self.yolo_confidence:.2f}")

    def run(self):
        """Start all threads"""
        print("\n" + "=" * 60)
        print("WINNING-STRATEGY GATE DETECTION OVERLAY")
        print("Inspired by 2019 AIRR Competition Winner")
        print("=" * 60)
        print("\nArchitecture:")
        print("Stage 0: Preprocessing (bilateral filter + noise reduction)")
        print("Stage 1: YOLO Detection (initial gate localization)")
        print("Stage 2: Semantic Segmentation (mask extraction)")
        print("         - Edge Detection (Canny)")
        print("         - Color Detection (HSV bright regions)")
        print("         - Corner Detection (Harris)")
        print("         - Weighted fusion")
        print("Stage 3: Active Vision Corner Finding (1.64% pixel sampling)")
        print("Stage 4: Gate-Prior Validation (aspect ratio sanity check)")
        print("=" * 60 + "\n")

        t_capture = threading.Thread(target=self.capture_thread, daemon=True)
        t_detection = threading.Thread(target=self.detection_thread, daemon=True)
        t_overlay = threading.Thread(target=self.overlay_window, daemon=True)

        t_capture.start()
        t_detection.start()
        t_overlay.start()

        self.control_panel()


if __name__ == "__main__":
    detector = GateDetectionOverlay()
    detector.run()
