#!/usr/bin/env python3
"""
YOLO-OBB Inference Utilities

Helper functions for using YOLO-OBB in the PnP workflow, including
angle normalization for symmetric components.

Supports both ultralytics YOLO and direct ONNX Runtime inference.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List
from pathlib import Path

# Try to import ultralytics (optional - can use ONNX Runtime directly)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

# Try to import ONNX Runtime (preferred for Raspberry Pi deployment)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


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


class ONNXComponentDetector:
    """
    Direct ONNX Runtime inference for YOLO-OBB models.
    
    Faster and more lightweight than ultralytics - ideal for Raspberry Pi.
    Does not require ultralytics package, only onnxruntime.
    
    Usage:
        detector = ONNXComponentDetector('models/best.onnx')
        result = detector.detect(frame, expected_angle=0)
        if result:
            print(f"Found at ({result['center_x']}, {result['center_y']}) angle={result['angle']}")
    """
    
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.5):
        """
        Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model (.onnx)
            imgsz: Inference image size (must match export size)
            conf: Confidence threshold
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Create ONNX Runtime session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use CPU execution provider (most compatible for Pi)
        providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        self.imgsz = imgsz
        self.conf = conf
        self.model_path = model_path
        
        print(f"ONNX model loaded: {model_path}")
        print(f"  Input: {self.input_name}")
        print(f"  Outputs: {self.output_names}")
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for YOLO inference.
        
        Args:
            frame: BGR image (numpy array)
        
        Returns:
            (preprocessed_tensor, scale, padding) for postprocessing
        """
        img_height, img_width = frame.shape[:2]
        
        # Calculate scale to fit in imgsz while maintaining aspect ratio
        scale = min(self.imgsz / img_width, self.imgsz / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize with aspect ratio
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_w = (self.imgsz - new_width) // 2
        pad_h = (self.imgsz - new_height) // 2
        padded = cv2.copyMakeBorder(resized, pad_h, self.imgsz - new_height - pad_h,
                                     pad_w, self.imgsz - new_width - pad_w,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to 0-1 and convert to NCHW format
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)   # Add batch dimension
        
        return tensor, scale, (pad_w, pad_h)
    
    def postprocess(self, outputs: List[np.ndarray], scale: float, 
                    padding: Tuple[int, int], original_shape: Tuple[int, int]) -> Optional[dict]:
        """
        Postprocess ONNX outputs to extract OBB detections.
        
        Args:
            outputs: Raw ONNX model outputs
            scale: Preprocessing scale factor
            padding: (pad_w, pad_h) from preprocessing
            original_shape: (height, width) of original image
        
        Returns:
            dict with detection info or None
        """
        # YOLO-OBB output format: [batch, num_predictions, 8 + 1 + num_classes]
        # where 8 = 4 corner points (x1,y1,x2,y2,x3,y3,x4,y4), 1 = angle, classes = confidence per class
        # Note: Actual format may vary by model version
        
        output = outputs[0]  # First output
        
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Transpose if needed (some models output [features, predictions])
        if output.shape[0] < output.shape[1]:
            output = output.T
        
        # Filter by confidence
        # For OBB, confidence is typically in a specific column
        # Format varies - this handles common YOLO-OBB export format
        
        best_conf = 0
        best_detection = None
        
        for det in output:
            # Try to extract confidence (usually after coordinates)
            # Common formats: [...coords..., conf, class] or [...coords..., class_confs...]
            if len(det) >= 9:
                # OBB format: x1,y1,x2,y2,x3,y3,x4,y4,conf or similar
                conf = float(det[8]) if len(det) > 8 else float(det[-1])
                
                if conf > self.conf and conf > best_conf:
                    best_conf = conf
                    best_detection = det
        
        if best_detection is None:
            return None
        
        # Extract corner points (first 8 values)
        corners = best_detection[:8].reshape(4, 2)
        
        # Remove padding and scale back to original image coordinates
        pad_w, pad_h = padding
        corners[:, 0] = (corners[:, 0] - pad_w) / scale
        corners[:, 1] = (corners[:, 1] - pad_h) / scale
        
        # Clip to image bounds
        orig_h, orig_w = original_shape
        corners[:, 0] = np.clip(corners[:, 0], 0, orig_w)
        corners[:, 1] = np.clip(corners[:, 1], 0, orig_h)
        
        # Calculate center
        center_x = float(corners[:, 0].mean())
        center_y = float(corners[:, 1].mean())
        
        # Calculate dimensions
        width = float(np.linalg.norm(corners[1] - corners[0]))
        height = float(np.linalg.norm(corners[2] - corners[1]))
        
        # Calculate angle from first edge
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        raw_angle = float(np.degrees(np.arctan2(dy, dx)))
        
        return {
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'angle': raw_angle,
            'raw_angle': raw_angle,
            'confidence': best_conf,
            'corners': corners
        }
    
    def detect(self, frame: np.ndarray, expected_angle: float = 0.0) -> Optional[dict]:
        """
        Detect component in frame using ONNX inference.
        
        Args:
            frame: BGR image (numpy array)
            expected_angle: Expected angle for normalization (degrees)
        
        Returns:
            dict with center_x, center_y, angle, confidence, etc.
            or None if no detection
        """
        original_shape = frame.shape[:2]
        
        # Preprocess
        tensor, scale, padding = self.preprocess(frame)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        
        # Postprocess
        result = self.postprocess(outputs, scale, padding, original_shape)
        
        if result is None:
            return None
        
        # Normalize angle to expected
        result['angle'] = normalize_angle_to_expected(result['raw_angle'], expected_angle)
        
        return result


class YOLOComponentDetector:
    """
    YOLO-OBB based component detector for PnP workflow.
    
    Wraps YOLO model with angle normalization for symmetric components.
    Supports both ultralytics (.pt/.onnx) and direct ONNX Runtime.
    
    Usage:
        detector = YOLOComponentDetector('models/best.onnx')
        result = detector.detect(frame, expected_angle=0)
        if result:
            print(f"Found at ({result['center_x']}, {result['center_y']}) angle={result['angle']}")
    """
    
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.5, 
                 use_onnx_runtime: bool = False):
        """
        Initialize detector.
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            imgsz: Inference image size (320 for speed, 640 for accuracy)
            conf: Confidence threshold
            use_onnx_runtime: If True, use direct ONNX Runtime (faster, requires .onnx model)
                             If False, use ultralytics (more features, slower)
        """
        self.imgsz = imgsz
        self.conf = conf
        self.model_path = model_path
        self.use_onnx_runtime = use_onnx_runtime
        
        model_ext = Path(model_path).suffix.lower()
        
        # Auto-select backend
        if use_onnx_runtime or (model_ext == '.onnx' and not ULTRALYTICS_AVAILABLE):
            if not ONNX_AVAILABLE:
                raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
            if model_ext != '.onnx':
                raise ValueError("ONNX Runtime requires .onnx model file")
            
            self._onnx_detector = ONNXComponentDetector(model_path, imgsz, conf)
            self.model = None
            self.use_onnx_runtime = True
            print(f"Using ONNX Runtime backend")
        else:
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError("ultralytics not installed. Run: pip install ultralytics")
            
            self.model = YOLO(model_path)
            self.model.overrides['conf'] = conf
            self.model.overrides['imgsz'] = imgsz
            self._onnx_detector = None
            self.use_onnx_runtime = False
            print(f"Using ultralytics backend")
    
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
        if self.use_onnx_runtime:
            return self._onnx_detector.detect(frame, expected_angle)
        else:
            results = self.model.predict(frame, verbose=False)
            return extract_obb_info(results, expected_angle)
    
    def detect_raw(self, frame):
        """
        Run raw YOLO inference without angle normalization.
        
        Returns YOLO results object for custom processing.
        Note: Only available when using ultralytics backend.
        """
        if self.use_onnx_runtime:
            raise NotImplementedError("detect_raw() not available with ONNX Runtime backend. Use detect() instead.")
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

