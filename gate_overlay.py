#!/usr/bin/env python3
"""
Gate detection overlay.

Screen-captures a monitor, runs the classical-CV gate detector from
gate_detection_core on each frame, and draws the detected gate polygons on a
transparent always-on-top overlay window. Detection parameters live in the
shared GateDetector and are loaded from gate_config.json on startup.
"""

import threading
import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL
from PIL import Image, ImageDraw, ImageTk, ImageGrab
import numpy as np
import cv2
import time

from gate_detection_core import GateDetector


class GateDetectionOverlay:
    def __init__(self, monitor_index=0):
        self.detector = GateDetector.from_config_file("gate_config.json")

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

        # Display settings (on the overlay object itself)
        self.show_labels = True
        self.show_centers = True

        # State tracking
        self.gates_passed = 0
        self.last_gate_center_y = None
        self.gate_age = {}  # Track age of each gate in frames, keyed by bbox

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

    def process_detections(self, frame):
        """Run the classical-CV gate detector on a BGR frame.

        Delegates entirely to the shared GateDetector so the detection thread
        is unchanged. Returns the list of gate dicts described in
        gate_detection_core (keys: corners, inner_corners, confidence, band, ...).
        """
        return self.detector.process_frame(frame)

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

    def _corner_points(self, corners):
        """Convert a 4x2 float ndarray of corners to a list of int (x, y) tuples."""
        points = []
        for pt in corners:
            points.append((int(round(float(pt[0]))), int(round(float(pt[1])))))
        return points

    def create_overlay_image(self):
        """Create transparent overlay with detection polygons"""
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        with self.lock:
            detections = self.current_detections.copy()

        for i, gate in enumerate(detections):
            conf = gate['confidence']

            if conf > 0.8:
                color = (0, 255, 0, 200)
            elif conf > 0.5:
                color = (255, 255, 0, 200)
            else:
                color = (255, 0, 0, 200)

            # Outer quad polygon; fall back to the bbox rectangle if absent.
            corners = gate.get('corners')
            if corners is not None:
                points = self._corner_points(corners)
                if len(points) >= 3:
                    draw.polygon(points, outline=color, width=2)
            else:
                x1, y1, x2, y2 = gate['x1'], gate['y1'], gate['x2'], gate['y2']
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Inner quad, drawn as a second, thinner polygon when present.
            inner = gate.get('inner_corners')
            if inner is not None:
                inner_points = self._corner_points(inner)
                if len(inner_points) >= 3:
                    draw.polygon(inner_points, outline=color, width=1)

            if self.show_centers:
                cx, cy = gate['center_x'], gate['center_y']
                draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)

            if self.show_labels:
                gate_key = (gate['x1'], gate['y1'], gate['x2'], gate['y2'])
                age = self.gate_age.get(gate_key, 0)
                text = f"G{i + 1} {conf:.2f} {gate['band']} (age:{age})"
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
        root.title("Gate Detector Control")
        root.geometry("420x600")

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

        ttk.Label(frame, text="Gate Detector", font=("Arial", 12, "bold")).pack()
        ttk.Label(frame, text="Classical CV (color-first pipeline)", font=("Arial", 9, "italic")).pack()
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        self.detection_button = ttk.Button(frame, text="DETECTION: ON", command=self.toggle_detection)
        self.detection_button.pack(fill='x', pady=5)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # Detection parameters (write straight onto the shared detector).
        ttk.Label(frame, text="Detection", font=("Arial", 10, "bold")).pack(anchor='w')

        ttk.Checkbutton(frame, text="Enable Preprocessing (bilateral filter)",
                        command=self.toggle_preprocess).pack(anchor='w')

        ttk.Label(frame, text="Bilateral Sigma", font=("Arial", 9)).pack(anchor='w')
        bilateral_scale = Scale(frame, from_=1, to=30, orient=HORIZONTAL,
                                command=lambda v: self.detector.update(bilateral_sigma=int(float(v))))
        bilateral_scale.set(self.detector.bilateral_sigma)
        bilateral_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Saturation Min", font=("Arial", 9)).pack(anchor='w')
        sat_scale = Scale(frame, from_=0, to=255, orient=HORIZONTAL,
                          command=lambda v: self.detector.update(sat_min=int(float(v))))
        sat_scale.set(self.detector.sat_min)
        sat_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Value Min (brightness)", font=("Arial", 9)).pack(anchor='w')
        val_scale = Scale(frame, from_=0, to=255, orient=HORIZONTAL,
                          command=lambda v: self.detector.update(val_min=int(float(v))))
        val_scale.set(self.detector.val_min)
        val_scale.pack(fill='x', pady=2)

        ttk.Label(frame, text="Gate Confidence Threshold", font=("Arial", 9)).pack(anchor='w')
        conf_scale = Scale(frame, from_=0.0, to=1.0, resolution=0.05, orient=HORIZONTAL,
                           command=lambda v: self.detector.update(confidence_threshold=float(v)))
        conf_scale.set(self.detector.confidence_threshold)
        conf_scale.pack(fill='x', pady=2)

        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(frame, text="Display Options", font=("Arial", 10, "bold")).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Labels", command=self.toggle_labels).pack(anchor='w')
        ttk.Checkbutton(frame, text="Show Centers", command=self.toggle_centers).pack(anchor='w')

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

    def toggle_preprocess(self):
        self.detector.update(preprocess_enabled=not self.detector.preprocess_enabled)

    def toggle_labels(self):
        self.show_labels = not self.show_labels

    def toggle_centers(self):
        self.show_centers = not self.show_centers

    def run(self):
        """Start all threads"""
        print("\n" + "=" * 60)
        print("GATE DETECTION OVERLAY (classical CV, color-first)")
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
