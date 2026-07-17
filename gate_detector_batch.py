#!/usr/bin/env python3
"""
Fast batch gate detector - processes video without display
Outputs JSON log with all detections and gate passage events
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np


class BatchGateDetector:
    def __init__(self, video_path, output_dir="logs"):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(str(self.video_path))

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.events = []
        self.gates_passed = 0
        self.last_gate_center_y = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"gate_analysis_{timestamp}.json"

    def detect_gate_passage(self, gates):
        """Detect when gate crosses from above-center to below-center"""
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

    def process_frame(self, frame, frame_idx, timestamp):
        """Run YOLO detection on single frame"""
        results = self.model(frame, conf=0.3, verbose=False)

        gates = []
        for det in results[0].boxes:
            conf = float(det.conf[0])
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            gates.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': w, 'height': h,
                'center_x': cx, 'center_y': cy,
                'confidence': conf
            })

        gates.sort(key=lambda g: g['height'], reverse=True)

        event = {
            'frame': frame_idx,
            'timestamp': timestamp,
            'gates_detected': len(gates),
            'gates': gates
        }

        if self.detect_gate_passage(gates):
            self.gates_passed += 1
            event['gate_passed'] = True
            event['total_gates_passed'] = self.gates_passed
            event['passage_confidence'] = gates[0]['confidence']

        self.events.append(event)
        return len(gates)

    def run(self):
        """Process entire video"""
        print(f"Processing: {self.video_path.name}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        print(f"Frames: {self.frame_count}")
        print()

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            time_sec = frame_idx / self.fps
            gates_count = self.process_frame(frame, frame_idx, time_sec)

            if (frame_idx + 1) % 100 == 0:
                progress = 100 * frame_idx / self.frame_count
                print(f"Frame {frame_idx + 1}/{self.frame_count} ({progress:.1f}%) - "
                      f"{gates_count} gates")

            frame_idx += 1

        self.cap.release()

        self.save_results()

    def save_results(self):
        """Write JSON log"""
        output = {
            'video': str(self.video_path.name),
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps,
            'total_frames': self.frame_count,
            'duration_sec': self.frame_count / self.fps,
            'total_gates_passed': self.gates_passed,
            'events': self.events
        }

        with open(self.log_file, 'w') as f:
            json.dump(output, f, indent=2)

        print()
        print(f"Complete: {self.log_file}")
        print(f"Total gates passed: {self.gates_passed}")
        print(f"Total detections: {sum(e['gates_detected'] for e in self.events)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gate_detector_batch.py <video_file>")
        print("Example: python gate_detector_batch.py video.mp4")
        sys.exit(1)

    video_file = sys.argv[1]

    if not Path(video_file).exists():
        print(f"Error: {video_file} not found")
        sys.exit(1)

    detector = BatchGateDetector(video_file)
    detector.run()
