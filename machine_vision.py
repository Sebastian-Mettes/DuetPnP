"""
Machine Vision Module for DuetPnP

This module handles all computer vision operations including:
- Camera configuration and coordinate transforms
- Component detection and template matching
- Tool position detection
- Window management and display
- HSV thresholding and circle detection

Optimized for Raspberry Pi 5 performance with template caching,
image downsampling, and efficient frame processing.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Callable
from collections import defaultdict
import json
import os
import sys

# Try to import YOLO (optional dependency)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None

# Try to import ONNX Runtime (optional - lighter weight alternative)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Add YOLO scripts to path for inference utilities
_yolo_scripts_path = os.path.join(os.path.dirname(__file__), 'YOLO', 'scripts')
if os.path.exists(_yolo_scripts_path) and _yolo_scripts_path not in sys.path:
    sys.path.insert(0, _yolo_scripts_path)

# Try to import our inference utilities (for ONNXComponentDetector)
try:
    from inference_utils import ONNXComponentDetector
    ONNX_DETECTOR_AVAILABLE = True
except ImportError:
    ONNXComponentDetector = None
    ONNX_DETECTOR_AVAILABLE = False


class CameraConfig:
    """
    Camera configuration handler for coordinate transforms and image processing.

    Handles camera-specific transformations including:
    - Image rotation and flipping for logical orientation
    - Pixel-to-machine coordinate transforms
    - Fixed camera offsets
    """

    def __init__(self, config_file: str):
        """
        Load camera configuration from JSON file.

        Args:
            config_file: Path to camera configuration JSON file

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Camera config file not found: {config_file}")

        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Validate required fields
        required_fields = ['camera_id', 'transform', 'fixed_offset']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in camera config: {field}")

        print(f"Loaded camera config: {self.config['name']}")

    def transform_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply flip/rotation to make image logically oriented.

        After transformation, all cameras show images where:
        - Object offset +X in image means move -X on machine
        - Object offset +Y in image means move -Y on machine

        Args:
            frame: Input BGR image

        Returns:
            Transformed BGR image
        """
        transformed = frame.copy()

        # Apply flip if specified
        flip_mode = self.config['transform'].get('image_flip')
        if flip_mode == 'vertical':
            transformed = cv2.flip(transformed, 0)
        elif flip_mode == 'horizontal':
            transformed = cv2.flip(transformed, 1)
        elif flip_mode == 'both':
            transformed = cv2.flip(transformed, -1)

        # Apply rotation if specified
        rotation = self.config['transform'].get('image_rotate', 0)
        if rotation == 90:
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            transformed = cv2.rotate(transformed, cv2.ROTATE_180)
        elif rotation == 270:
            transformed = cv2.rotate(transformed, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return transformed

    def pixel_to_machine_movement(self, pixel_x_offset: float, pixel_y_offset: float) -> Tuple[float, float]:
        """
        Convert pixel offsets to machine movement in mm.

        Args:
            pixel_x_offset: Horizontal pixel offset (positive = right in image)
            pixel_y_offset: Vertical pixel offset (positive = down in image)

        Returns:
            (machine_x_move, machine_y_move) in mm
        """
        t = self.config['transform']

        # Apply transformation matrix
        machine_x = (pixel_x_offset * t['pixel_x_to_machine_x'] +
                     pixel_y_offset * t['pixel_y_to_machine_x']) * t['pixel_to_mm']
        machine_y = (pixel_x_offset * t['pixel_x_to_machine_y'] +
                     pixel_y_offset * t['pixel_y_to_machine_y']) * t['pixel_to_mm']

        return (machine_x, machine_y)

    def get_fixed_offset(self) -> Tuple[float, float]:
        """Get fixed camera offset from machine origin."""
        offset = self.config['fixed_offset']
        return (offset['x'], offset['y'])

    def get_led_pin(self) -> str:
        """Get LED control pin name."""
        return self.config.get('led_control', {}).get('pin', 'fan0')


class VisionTools:
    """
    Computer vision operations for component detection and tool calibration.

    Provides methods for:
    - Frame capture with optimizations
    - Template matching for component detection
    - Circle detection for tool finding
    - HSV color filtering
    - Tool-specific vision parameter loading
    """

    def __init__(self, camera_number: int, target: str = 'tool', tool_number: Optional[int] = None,
                 camera_config: Optional[CameraConfig] = None, debug: bool = False):
        """
        Initialize VisionTools with camera and optional configuration.

        Args:
            camera_number: Camera device number (0, 2, etc.)
            target: Target type ('tool', 'camera', 'target')
            tool_number: Specific tool number for tool-specific vision params (optional)
            camera_config: CameraConfig instance for coordinate transforms (optional)
            debug: Enable verbose debug output (default False)
        """
        self.camera_number = camera_number
        self.target = target
        self.tool_number = tool_number
        self.camera_config = camera_config
        self.debug = debug

        # Initialize camera with V4L2 backend for Raspberry Pi
        self.camera = cv2.VideoCapture(self.camera_number, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_number}")

        # Optimize camera buffer for low latency (Raspberry Pi optimization)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

        # Set resolution (1280x960 preferred)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        # Verify resolution
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != 1280 or actual_height != 960:
            print(f"Warning: Could not set 1280x960. Actual: {actual_width}x{actual_height}")
            # Try fallback resolutions
            if actual_width < 1280:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Get final camera resolution (raw from camera)
        self.raw_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.raw_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate actual image dimensions after transformation
        # 90° or 270° rotation swaps width and height
        if camera_config:
            rotation = camera_config.config['transform'].get('image_rotate', 0)
            if rotation in [90, 270]:
                self.width = self.raw_height
                self.height = self.raw_width
            else:
                self.width = self.raw_width
                self.height = self.raw_height
        else:
            self.width = self.raw_width
            self.height = self.raw_height

        self.IMAGE_CENTER = (self.width // 2, self.height // 2)
        print(f"Camera {camera_number} initialized at {self.raw_width}x{self.raw_height} (transformed: {self.width}x{self.height})")

        # HSV threshold values (defaults)
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([180, 255, 255])

        # Region of interest for faster searching (None = full image)
        # Format: (x, y, width, height) defining center and size of search region
        self.search_roi = None

        # Circle detection parameters
        self.dp = 1
        self.min_dist = 20
        self.param1 = 100
        self.param2 = 30
        self.min_radius = 0
        self.max_radius = 0

        # Circle tracking
        self.circle_history = defaultdict(int)
        self.last_circles = None
        self.min_frames = 1

        # Template cache for performance (avoids disk I/O)
        self._template_cache = {}

        # Current frame storage
        self.frame = None
        self.is_component_detected = False

        # YOLO model (lazy loaded)
        self._yolo_model = None
        self._yolo_model_path = None
        self._yolo_imgsz = 320  # Default inference size for speed
        self._yolo_conf = 0.5   # Default confidence threshold
        self._use_onnx_runtime = False  # Use ONNX Runtime if available
        self._onnx_detector = None  # ONNXComponentDetector instance

        # Training data capture settings (for ML model improvement)
        self._training_capture_enabled = False
        self._training_capture_dir = None
        self._training_capture_count = 0
        self._training_capture_session_id = None
        self._training_capture_min_conf = 0.5  # Minimum confidence to save

    def capture_frame(self, clear_buffer: bool = False) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera.

        Args:
            clear_buffer: If True, clear camera buffer before capturing (use sparingly)

        Returns:
            BGR image frame or None if capture failed
        """
        if self.camera is None:
            print("Camera not initialized")
            return None

        # Only clear buffer if explicitly requested (after long pause)
        if clear_buffer:
            for _ in range(3):
                self.camera.read()

        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None

        # Apply camera transform if config available
        if self.camera_config:
            frame = self.camera_config.transform_image(frame)

        self.frame = frame
        return frame

    def get_image_dimensions(self) -> Tuple[int, int]:
        """Get camera frame dimensions (width, height)."""
        return (self.width, self.height)

    def get_image_center(self) -> Tuple[int, int]:
        """Get image center coordinates."""
        return self.IMAGE_CENTER

    def set_search_roi(self, width_fraction: float = 0.25, height_fraction: float = 0.25):
        """
        Set a reduced search region of interest centered on the image.
        
        Args:
            width_fraction: Fraction of image width to search (0.25 = 25%)
            height_fraction: Fraction of image height to search (0.25 = 25%)
        """
        roi_width = int(self.width * width_fraction)
        roi_height = int(self.height * height_fraction)
        roi_x = (self.width - roi_width) // 2
        roi_y = (self.height - roi_height) // 2
        self.search_roi = (roi_x, roi_y, roi_width, roi_height)
        if self.debug:
            print(f"Search ROI set: {self.search_roi} ({width_fraction*100:.0f}% x {height_fraction*100:.0f}%)")

    def clear_search_roi(self):
        """Clear the search ROI to search the full image."""
        if self.search_roi is not None and self.debug:
            print("Search ROI cleared - using full image")
        self.search_roi = None

    def _apply_roi_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Apply ROI mask to frame if set, returning masked frame and offset.
        
        Returns:
            (cropped_frame, (x_offset, y_offset)) where offset is the top-left corner of ROI
        """
        if self.search_roi is None:
            return frame, (0, 0)
        
        x, y, w, h = self.search_roi
        # Ensure bounds are valid
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        return frame[y:y+h, x:x+w], (x, y)

    def find_component(self, template_path: str, angle: int = 0, exact_angle: bool = False) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
        """
        Find component in camera frame using template matching.
        Uses coarse-to-fine search for better performance.

        Args:
            template_path: Path to template image file
            angle: Expected angle offset in degrees
            exact_angle: If True, only check at the specified angle (no angular search).
                        Much faster when component orientation is already known.

        Returns:
            ((center_x, center_y), rotation_angle) if found, (None, None) otherwise
        """
        if self.frame is None:
            return (None, None)

        # Load template with caching
        self.search_frame = self.frame.copy()
        if template_path not in self._template_cache:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise ValueError(f"Could not load template: {template_path}")
            self._template_cache[template_path] = template
        else:
            template = self._template_cache[template_path]

        # Apply ROI if set (for faster searching when target is near center)
        roi_offset = (0, 0)
        if self.search_roi is not None:
            self.search_frame, roi_offset = self._apply_roi_mask(self.search_frame)

        image_height, image_width = self.search_frame.shape[:2]
        old_center = self.IMAGE_CENTER
        self.IMAGE_CENTER = (self.width // 2, self.height // 2)  # Always use full image center
        if self.debug:
            print(f"DEBUG find_component: Frame dimensions: {image_width}x{image_height}, IMAGE_CENTER: {self.IMAGE_CENTER}, ROI offset: {roi_offset}")
            if old_center != self.IMAGE_CENTER:
                print(f"⚠️  WARNING: IMAGE_CENTER changed from {old_center} to {self.IMAGE_CENTER}")

        # Downsample for faster search
        scale_factor = 0.5
        gray = cv2.cvtColor(self.search_frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_AREA)

        # Cache downsampled template
        template_small_key = f"{template_path}_small"
        if template_small_key not in self._template_cache:
            template_small = cv2.resize(template, None, fx=scale_factor, fy=scale_factor,
                                       interpolation=cv2.INTER_AREA)
            self._template_cache[template_small_key] = template_small
        else:
            template_small = self._template_cache[template_small_key]

        best_match = None
        best_score = -1
        best_angle = 0
        best_template_shape = None

        # Stage 1: Coarse search (5-degree steps, downsampled)
        # If exact_angle=True, only check the specified angle (much faster!)
        if exact_angle:
            coarse_angles = [angle]  # Single angle only
        else:
            coarse_angles = list(range(-10 + angle, 10 + angle, 1))  # Full search

        coarse_best_angle = 0
        coarse_best_score = -1

        for test_angle in coarse_angles:
            h, w = template_small.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), test_angle, 1.0)

            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            matrix[0, 2] += (new_w / 2) - (w / 2)
            matrix[1, 2] += (new_h / 2) - (h / 2)

            rotated = cv2.warpAffine(template_small, matrix, (new_w, new_h))
            result = cv2.matchTemplate(gray_small, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > coarse_best_score:
                coarse_best_score = max_val
                coarse_best_angle = test_angle

            if max_val > 0.99:  # Early termination
                break

        # Stage 2: Fine search (2-degree steps, full resolution)
        # If exact_angle=True, skip fine search (already checked exact angle)
        if exact_angle:
            fine_angles = [angle]  # Single angle only
        else:
            fine_start = coarse_best_angle - 6
            fine_end = coarse_best_angle + 6
            fine_angles = range(fine_start, fine_end + 1, 2)

        for test_angle in fine_angles:
            h, w = template.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), test_angle, 1.0)

            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            matrix[0, 2] += (new_w / 2) - (w / 2)
            matrix[1, 2] += (new_h / 2) - (h / 2)

            rotated = cv2.warpAffine(template, matrix, (new_w, new_h))
            result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_match = max_loc
                best_angle = test_angle
                best_template_shape = rotated.shape

            if max_val > 0.95:  # Early termination
                break

        # Check if match is good enough
        if best_score > 0.6:
            h, w = best_template_shape
            # TEMPORARY TEST: Try different offset multipliers
            # Normal: center = (best_match[0] + w//2, best_match[1] + h//2)
            # Test 0x: center = (best_match[0] + 0, best_match[1] + 0)  # No offset
            # Test 2x: center = (best_match[0] + w, best_match[1] + h)  # Double offset
            center = (best_match[0] + w//2, best_match[1] + h//2)  # NORMAL (1x offset)
            # Add ROI offset back to get coordinates in full image space
            center = (center[0] + roi_offset[0], center[1] + roi_offset[1])
            if self.debug:
                print(f"DEBUG template_match: best_match top-left=({best_match[0]}, {best_match[1]}), template_size=({w}x{h}), adding offset=({w//2}, {h//2}), ROI offset={roi_offset}, final_center={center}")
            
            # Save training sample if enabled (use original template dimensions)
            if self._training_capture_enabled:
                orig_h, orig_w = template.shape[:2]
                corners = self._calculate_obb_corners(
                    center[0], center[1], orig_w, orig_h, -best_angle
                )
                self._save_training_sample(self.frame, corners, best_score)
            
            self.is_component_detected = True
            return (center, best_angle)
        else:
            self.is_component_detected = False
            return (None, None)

    def load_yolo_model(self, model_path: str, imgsz: int = 320, conf: float = 0.5,
                        use_onnx_runtime: bool = None):
        """
        Load YOLO-OBB model for component detection.
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
            imgsz: Inference image size (320 for speed, 640 for accuracy)
            conf: Confidence threshold (default 0.5)
            use_onnx_runtime: If True, use ONNX Runtime (faster, requires .onnx model)
                             If False, use ultralytics
                             If None, auto-select based on availability
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        model_ext = os.path.splitext(model_path)[1].lower()
        
        # Auto-select backend
        if use_onnx_runtime is None:
            # Prefer ONNX Runtime for .onnx models if available
            if model_ext == '.onnx' and ONNX_DETECTOR_AVAILABLE:
                use_onnx_runtime = True
            elif YOLO_AVAILABLE:
                use_onnx_runtime = False
            elif ONNX_DETECTOR_AVAILABLE and model_ext == '.onnx':
                use_onnx_runtime = True
            else:
                raise ImportError(
                    "No YOLO backend available. Install one of:\n"
                    "  pip install ultralytics  (full featured)\n"
                    "  pip install onnxruntime  (lightweight, .onnx only)"
                )
        
        self._yolo_model_path = model_path
        self._yolo_imgsz = imgsz
        self._yolo_conf = conf
        self._use_onnx_runtime = use_onnx_runtime
        
        print(f"Loading YOLO model: {model_path}")
        print(f"Backend: {'ONNX Runtime' if use_onnx_runtime else 'ultralytics'}")
        
        if use_onnx_runtime:
            if not ONNX_DETECTOR_AVAILABLE:
                raise ImportError("ONNX Runtime detector not available. Check onnxruntime installation.")
            if model_ext != '.onnx':
                raise ValueError("ONNX Runtime requires .onnx model file")
            
            self._onnx_detector = ONNXComponentDetector(model_path, imgsz, conf)
            self._yolo_model = None
            print(f"YOLO model loaded via ONNX Runtime (imgsz={imgsz}, conf={conf})")
        else:
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics not installed. Run: pip install ultralytics")
            
            self._yolo_model = YOLO(model_path)
            self._yolo_model.overrides['conf'] = conf
            self._yolo_model.overrides['imgsz'] = imgsz
            self._onnx_detector = None
            print(f"YOLO model loaded via ultralytics (imgsz={imgsz}, conf={conf})")

    def find_component_yolo(self, expected_angle: float = 0.0) -> Tuple[Optional[Tuple[int, int]], Optional[float]]:
        """
        Find component in camera frame using YOLO-OBB model.
        
        Much faster than template matching (~50-100ms vs 500ms+).
        Returns same format as find_component() for drop-in replacement.
        
        Supports both ultralytics and ONNX Runtime backends.
        
        Args:
            expected_angle: Expected component angle for normalization.
                           Used to handle 90°/180° ambiguity in symmetric components.
        
        Returns:
            ((center_x, center_y), angle) if found, (None, None) otherwise
        """
        if self._yolo_model is None and self._onnx_detector is None:
            print("YOLO model not loaded. Call load_yolo_model() first.")
            return (None, None)
        
        if self.frame is None:
            print("No frame captured. Call capture_frame() first.")
            return (None, None)
        
        # Use ONNX Runtime backend if available
        if self._use_onnx_runtime and self._onnx_detector is not None:
            result = self._onnx_detector.detect(self.frame, expected_angle)
            
            if result is None:
                self.is_component_detected = False
                return (None, None)
            
            # Save training sample if enabled
            if self._training_capture_enabled and 'corners' in result:
                self._save_training_sample(
                    self.frame, 
                    result['corners'], 
                    result.get('confidence', 1.0)
                )
            
            self.is_component_detected = True
            return ((int(result['center_x']), int(result['center_y'])), result['angle'])
        
        # Fallback to ultralytics
        if self._yolo_model is None:
            print("No YOLO model available.")
            return (None, None)
        
        # Run inference
        results = self._yolo_model.predict(self.frame, verbose=False)
        
        # Check for detections
        if results is None or len(results) == 0:
            self.is_component_detected = False
            return (None, None)
        
        if results[0].obb is None or len(results[0].obb.xyxyxyxy) == 0:
            self.is_component_detected = False
            return (None, None)
        
        obb = results[0].obb
        
        # Get the highest confidence detection
        best_idx = int(obb.conf.argmax())
        corners = obb.xyxyxyxy[best_idx].cpu().numpy()
        
        # Calculate center
        center_x = float(corners[:, 0].mean())
        center_y = float(corners[:, 1].mean())
        
        # Calculate raw angle from first edge
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        raw_angle = float(np.degrees(np.arctan2(dy, dx)))
        
        # Normalize angle to be within ±45° of expected
        # Handles 90°/180° ambiguity for symmetric components
        normalized_angle = self._normalize_angle_to_expected(raw_angle, expected_angle)
        
        # Save training sample if enabled
        if self._training_capture_enabled:
            confidence = float(obb.conf[best_idx].cpu().numpy())
            self._save_training_sample(self.frame, corners, confidence)
        
        self.is_component_detected = True
        return ((int(center_x), int(center_y)), normalized_angle)

    def _normalize_angle_to_expected(self, predicted_angle: float, expected_angle: float, 
                                      tolerance: float = 45.0) -> float:
        """
        Normalize predicted angle to be within ±tolerance of expected angle.
        
        For rectangular components, YOLO might use a different edge as reference,
        causing 90° or 180° offsets. This picks the angle closest to expected.
        """
        # Try all 90° rotations
        candidates = [
            predicted_angle,
            predicted_angle + 90,
            predicted_angle - 90,
            predicted_angle + 180,
            predicted_angle - 180,
            predicted_angle + 270,
            predicted_angle - 270,
        ]
        
        best_angle = predicted_angle
        best_diff = float('inf')
        
        for candidate in candidates:
            # Normalize to -180 to 180 range
            normalized = ((candidate + 180) % 360) - 180
            
            # Calculate difference from expected
            diff = abs(normalized - expected_angle)
            if diff > 180:
                diff = 360 - diff
            
            if diff < best_diff:
                best_diff = diff
                best_angle = normalized
        
        return best_angle

    def enable_training_capture(self, output_dir: str = None, min_confidence: float = 0.5,
                                  session_id: str = None):
        """
        Enable automatic saving of successful detections for ML training.
        
        Saves images and YOLO-OBB format labels when components are detected
        during normal operation. This creates passive training data collection.
        
        Args:
            output_dir: Directory to save images/labels (default: YOLO/data)
            min_confidence: Minimum detection confidence to save (default: 0.5)
            session_id: Session identifier for filenames (default: auto-generated)
        """
        from datetime import datetime
        
        if output_dir is None:
            # Default to YOLO data directory relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, 'YOLO', 'data')
        
        # Create images and labels directories
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        self._training_capture_enabled = True
        self._training_capture_dir = output_dir
        self._training_capture_min_conf = min_confidence
        self._training_capture_count = 0
        self._training_capture_session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Training capture enabled: {output_dir}")
        print(f"  Session: {self._training_capture_session_id}, min_conf: {min_confidence}")

    def disable_training_capture(self):
        """Disable automatic training data capture."""
        if self._training_capture_enabled:
            print(f"Training capture disabled. Saved {self._training_capture_count} samples.")
        self._training_capture_enabled = False

    def _calculate_obb_corners(self, center_x: float, center_y: float,
                                 width: float, height: float, angle_deg: float) -> np.ndarray:
        """
        Calculate OBB corners from center, dimensions, and angle.
        
        Args:
            center_x, center_y: Center of the box in pixels
            width, height: Dimensions of the box (before rotation)
            angle_deg: Rotation angle in degrees (counterclockwise)
        
        Returns:
            numpy array of shape (4, 2) with corner coordinates
        """
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Half dimensions
        hw = width / 2
        hh = height / 2
        
        # Corner offsets from center (before rotation)
        # Top-left, Top-right, Bottom-right, Bottom-left
        corners_local = [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh)
        ]
        
        # Rotate and translate to image coordinates
        corners = []
        for dx, dy in corners_local:
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            corners.append([center_x + rx, center_y + ry])
        
        return np.array(corners)

    def _save_training_sample(self, frame: np.ndarray, corners: np.ndarray, 
                               confidence: float):
        """
        Save a training sample (image + YOLO-OBB label).
        
        Args:
            frame: BGR image (numpy array)
            corners: OBB corners array shape (4, 2) with pixel coordinates
            confidence: Detection confidence (for filtering)
        """
        if not self._training_capture_enabled:
            return
        
        if confidence < self._training_capture_min_conf:
            return
        
        if frame is None or corners is None:
            return
        
        # Get image dimensions
        img_height, img_width = frame.shape[:2]
        
        # Generate filename
        filename = f"{self._training_capture_session_id}_{self._training_capture_count:04d}"
        images_dir = os.path.join(self._training_capture_dir, 'images')
        labels_dir = os.path.join(self._training_capture_dir, 'labels')
        
        img_path = os.path.join(images_dir, f"{filename}.jpg")
        label_path = os.path.join(labels_dir, f"{filename}.txt")
        
        # Save image
        cv2.imwrite(img_path, frame)
        
        # Convert corners to YOLO-OBB format (normalized 0-1)
        # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
        normalized_coords = []
        for i in range(4):
            nx = float(corners[i, 0]) / img_width
            ny = float(corners[i, 1]) / img_height
            # Clamp to valid range
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            normalized_coords.extend([nx, ny])
        
        label_str = f"0 {' '.join(f'{v:.6f}' for v in normalized_coords)}"
        
        with open(label_path, 'w') as f:
            f.write(label_str + '\n')
        
        self._training_capture_count += 1
        
        # Periodic status update
        if self._training_capture_count % 50 == 0:
            print(f"  [Training] Saved {self._training_capture_count} samples")

    def find_tool_position(self, downsample: bool = True) -> Optional[Tuple[int, int]]:
        """
        Locate tool or target in camera frame using saved HSV and circle parameters.
        Optimized for speed with optional downsampling.

        Args:
            downsample: If True, process at 50% scale for speed

        Returns:
            (x, y) pixel coordinates or None if not detected
        """
        # Load parameters from file
        if self.tool_number is not None:
            params_file = f"config/vision_params_camera{self.camera_number}_tool{self.tool_number}.json"
        else:
            params_file = f"config/vision_params_camera{self.camera_number}_{self.target}.json"

        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
        except FileNotFoundError:
            print(f"Vision parameters not found: {params_file}")
            print("Run vision parameter calibration first.")
            return None

        # Extract parameters
        hsv_lower = np.array(params['hsv_lower'])
        hsv_upper = np.array(params['hsv_upper'])
        circle_params = params['circle_params']

        # Downsampling factor
        scale = 0.5 if downsample else 1.0

        frame_count = 0
        max_attempts = 30  # Don't loop forever

        while frame_count < max_attempts:
            frame = self.capture_frame()
            if frame is None:
                continue

            # Apply ROI if set (for faster searching when target is near center)
            roi_offset = (0, 0)
            if self.search_roi is not None:
                frame, roi_offset = self._apply_roi_mask(frame)

            # Downsample for speed
            if downsample:
                frame_small = cv2.resize(frame, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            # Process: HSV -> mask -> gray with mask
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_and(gray, gray, mask=mask)

            # Detect circles with scaled parameters
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=circle_params['dp'],
                minDist=int(circle_params['minDist'] * scale),
                param1=circle_params['param1'],
                param2=circle_params['param2'],
                minRadius=int(circle_params['minRadius'] * scale),
                maxRadius=int(circle_params['maxRadius'] * scale)
            )

            if circles is not None:
                circles = np.int32(np.around(circles))

                if len(circles[0]) == 1:
                    # Single circle found - scale back to original coordinates and add ROI offset
                    x = int(circles[0][0][0] / scale) + roi_offset[0]
                    y = int(circles[0][0][1] / scale) + roi_offset[1]
                    return (x, y)

                elif len(circles[0]) > 1:
                    # Multiple circles - check if close together
                    max_distance = 0
                    for i in range(len(circles[0])):
                        for j in range(i + 1, len(circles[0])):
                            dist = np.sqrt((circles[0][i][0] - circles[0][j][0])**2 +
                                         (circles[0][i][1] - circles[0][j][1])**2)
                            max_distance = max(max_distance, dist)

                    if max_distance <= (10 * scale):
                        # Circles close together - take average, add ROI offset
                        avg_x = int(np.mean([c[0] for c in circles[0]]) / scale) + roi_offset[0]
                        avg_y = int(np.mean([c[1] for c in circles[0]]) / scale) + roi_offset[1]
                        return (avg_x, avg_y)

            frame_count += 1

        print("Could not detect tool after maximum attempts")
        return None

    def cleanup(self):
        """Release camera resources."""
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            print(f"Camera {self.camera_number} released")

    def __del__(self):
        """Ensure camera is released on deletion."""
        self.cleanup()


# Window management functions

def create_display_window(window_name: str = 'PnP Camera View', fullscreen: bool = False):
    """
    Create an OpenCV window for displaying camera images.

    Args:
        window_name: Name of the window
        fullscreen: If True, create fullscreen window
    """
    if fullscreen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


def display_image(image: np.ndarray, window_name: str = 'PnP Camera View',
                 text: Optional[str] = None, wait_key: int = 1):
    """
    Display image in window with optional text overlay.

    Args:
        image: BGR image to display
        window_name: Name of window to update
        text: Optional text to overlay
        wait_key: Milliseconds to wait for key press (1 = minimal delay)
    """
    display_img = image.copy()

    if text:
        cv2.putText(display_img, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name, display_img)
    cv2.waitKey(wait_key)


def show_frame_with_overlay(frame: np.ndarray, detected_pos: Optional[Tuple[int, int]] = None,
                            center_pos: Optional[Tuple[int, int]] = None,
                            window_name: str = 'Detection', text: Optional[str] = None):
    """
    Show frame with detection visualization overlay.

    Args:
        frame: BGR image
        detected_pos: (x, y) of detected target
        center_pos: (x, y) of image center crosshair
        window_name: Window name
        text: Optional status text
    """
    display = frame.copy()

    # Draw detected position
    if detected_pos:
        x, y = detected_pos
        cv2.circle(display, (x, y), 20, (0, 255, 0), 3)
        cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

    # Draw center crosshair
    if center_pos:
        cx, cy = center_pos
        cv2.line(display, (cx-20, cy), (cx+20, cy), (255, 0, 0), 2)
        cv2.line(display, (cx, cy-20), (cx, cy+20), (255, 0, 0), 2)

    # Add text
    if text:
        cv2.putText(display, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Create named window if it doesn't exist (helps on some systems like Raspberry Pi)
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except:
        pass

    cv2.imshow(window_name, display)
    cv2.waitKey(1)


def load_camera_config(camera_number: int) -> CameraConfig:
    """
    Load camera configuration from file.

    Args:
        camera_number: Camera ID (0, 2, etc.)

    Returns:
        CameraConfig instance
    """
    try:
        config_file = f"config/camera_config_{camera_number}.json"
    except:
        print('camera config not file, trying one foler up')
        try:
            config_file = f"../config/camera_config_{camera_number}.json"
        except:
            print('camera config file not found!')
    return CameraConfig(config_file)
