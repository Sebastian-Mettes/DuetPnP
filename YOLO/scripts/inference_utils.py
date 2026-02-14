#!/usr/bin/env python3
"""
YOLO-OBB Inference Utilities

Helper functions for using YOLO-OBB in the PnP workflow, including
angle normalization for symmetric components.
"""

import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def normalize_angle_to_expected(predicted_angle: float, expected_angle: float, 
                                 tolerance: float = 45.0) -> float:
    """
    Normalize predicted angle to be within ±tolerance of expected angle.
    
    For rectangular components, YOLO might use a different edge as reference,
    causing 90° or 180° offsets. This function picks the angle closest to expected
    by trying all 90° rotations.
    
    Args:
        predicted_angle: Angle from YOLO prediction (degrees)
        expected_angle: Expected angle based on workflow context (degrees)
        tolerance: Maximum allowed deviation from expected (default: 45°)
    
    Returns:
        Normalized angle within ±tolerance of expected_angle
    """
    # Candidate angles: try all 90° rotations
    candidates = [
        predicted_angle,
        predicted_angle + 90,
        predicted_angle - 90,
        predicted_angle + 180,
        predicted_angle - 180,
        predicted_angle + 270,
        predicted_angle - 270,
    ]
    
    # Find the candidate closest to expected
    best_angle = predicted_angle
    best_diff = float('inf')
    
    for candidate in candidates:
        # Normalize candidate to -180 to 180 range
        normalized = ((candidate + 180) % 360) - 180
        
        # Calculate difference from expected
        diff = abs(normalized - expected_angle)
        if diff > 180:
            diff = 360 - diff
        
        if diff < best_diff:
            best_diff = diff
            best_angle = normalized
    
    return best_angle


def extract_obb_info(results, expected_angle: float = 0.0) -> Optional[dict]:
    """
    Extract OBB information from YOLO results with angle normalization.
    
    Args:
        results: YOLO prediction results
        expected_angle: Expected component angle for normalization
    
    Returns:
        dict with keys: center_x, center_y, width, height, angle, confidence
        or None if no detection
    """
    if results is None or len(results) == 0:
        return None
    
    if results[0].obb is None or len(results[0].obb.xyxyxyxy) == 0:
        return None
    
    obb = results[0].obb
    
    # Get the highest confidence detection
    best_idx = obb.conf.argmax()
    corners = obb.xyxyxyxy[best_idx].cpu().numpy()
    confidence = float(obb.conf[best_idx])
    
    # Calculate center
    center_x = corners[:, 0].mean()
    center_y = corners[:, 1].mean()
    
    # Calculate dimensions (distance between adjacent corners)
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[2] - corners[1])
    
    # Calculate raw angle from first edge
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    raw_angle = np.degrees(np.arctan2(dy, dx))
    
    # Normalize angle to be within ±45° of expected
    normalized_angle = normalize_angle_to_expected(raw_angle, expected_angle)
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'width': width,
        'height': height,
        'angle': normalized_angle,
        'raw_angle': raw_angle,
        'confidence': confidence,
        'corners': corners
    }


class YOLOComponentDetector:
    """
    YOLO-OBB based component detector for PnP workflow.
    
    Wraps YOLO model with angle normalization for symmetric components.
    
    Usage:
        detector = YOLOComponentDetector('models/best.onnx')
        result = detector.detect(frame, expected_angle=0)
        if result:
            print(f"Found at ({result['center_x']}, {result['center_y']}) angle={result['angle']}")
    """
    
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.5):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            imgsz: Inference image size (320 for speed, 640 for accuracy)
            conf: Confidence threshold
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.model.overrides['conf'] = conf
        self.model.overrides['imgsz'] = imgsz
        self.imgsz = imgsz
        self.conf = conf
    
    def detect(self, frame, expected_angle: float = 0.0) -> Optional[dict]:
        """
        Detect component in frame.
        
        Args:
            frame: BGR image (numpy array)
            expected_angle: Expected angle for normalization (degrees)
        
        Returns:
            dict with center_x, center_y, angle, confidence, etc.
            or None if no detection
        """
        results = self.model.predict(frame, verbose=False)
        return extract_obb_info(results, expected_angle)
    
    def detect_raw(self, frame):
        """
        Run raw YOLO inference without angle normalization.
        
        Returns YOLO results object for custom processing.
        """
        return self.model.predict(frame, verbose=False)


# Example usage and testing
if __name__ == '__main__':
    print("Testing angle normalization (handles 90° and 180° offsets):\n")
    
    test_cases = [
        (0, 0),      # Predicted 0, expected 0 → should be 0
        (90, 0),     # Predicted 90, expected 0 → should be 0 (90° offset)
        (-90, 0),    # Predicted -90, expected 0 → should be 0 (90° offset)
        (180, 0),    # Predicted 180, expected 0 → should be 0 (180° symmetric)
        (-180, 0),   # Predicted -180, expected 0 → should be 0
        (85, 0),     # Predicted 85, expected 0 → should be -5 (close to 90° offset)
        (-85, 0),    # Predicted -85, expected 0 → should be 5 (close to -90° offset)
        (170, 0),    # Predicted 170, expected 0 → should be -10 (close to 180°)
        (45, 30),    # Predicted 45, expected 30 → should be 45
        (135, 30),   # Predicted 135, expected 30 → should be 45 (90° offset from 30)
    ]
    
    print(f"{'Predicted':>10} | {'Expected':>8} | {'Normalized':>10} | {'Diff':>6}")
    print("-" * 45)
    for predicted, expected in test_cases:
        result = normalize_angle_to_expected(predicted, expected)
        diff = abs(result - expected)
        if diff > 180:
            diff = 360 - diff
        print(f"{predicted:>10}° | {expected:>8}° | {result:>10.1f}° | {diff:>5.1f}°")

