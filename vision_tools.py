import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import defaultdict
import json
import os
import sys

class VisionTools:
    """
    A class to handle computer vision operations for toolhead calibration.
    
    This class provides methods to capture and process camera images,
    detect tools, and perform calibration-relatedd vision tasks.
    
    Attributes:
        camera: OpenCV video capture object
        image_width (int): Width of camera frame in pixels
        image_height (int): Height of camera frame in pixels
        hsv_lower (np.array): Lower bounds for HSV color filtering
        hsv_upper (np.array): Upper bounds for HSV color filtering
    """

    def __init__(self,camera_number,target = 'tool'):
        """
        Initialize VisionTools with camera and default parameters.

        Args:
            camera_number (int): Camera device number
            target (str): Target type - either 'tool' or 'camera'
        """
        self.camera_number = camera_number
        self.target = target
        self.camera = cv2.VideoCapture(self.camera_number,cv2.CAP_V4L2)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

        # Optimize camera buffer for low latency (Raspberry Pi optimization)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

        # Set 720p resolution (1280x720)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        # Verify resolution was set correctly
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != 1280 or actual_height != 960:
            print(f"Warning: Could not set 720p resolution. Actual resolution: {actual_width}x{actual_height}")
            # Try to set the closest supported resolution
            if actual_width < 1280:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Get final camera resolution
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_component_detected = False
        # Initialize IMAGE_CENTER for backward compatibility
        self.IMAGE_CENTER = (self.width // 2, self.height // 2)
        print(f"Camera initialized at {self.width}x{self.height} resolution")

        # HSV threshold values
        self.hsv_lower = np.array([0, 0, 0])
        self.hsv_upper = np.array([180, 255, 255])

        # Circle detection parameters
        self.dp = 1
        self.min_dist = 20
        self.param1 = 100
        self.param2 = 30
        self.min_radius = 0
        self.max_radius = 0

        # Circle tracking
        self.circle_history = defaultdict(int)  # Track how many frames each circle has been seen
        self.last_circles = None  # Store last detected circles
        self.min_frames = 1  # Minimum number of frames a circle must appear in

        # Camera settings
        self.brightness = 0
        self.contrast = 0

        # Template cache for performance
        self._template_cache = {}  # Cache loaded templates to avoid disk I/O

    
    def set_fixed_camera_offset(self, x: float, y: float):
        """
        Set the fixed camera offset.
        Stores both as dict (new format) and list (legacy format) for compatibility.
        """
        # Legacy format for backward compatibility
        self.camera_offset = [x, y]
        # Also set CAMERA_OFFSET for backward compatibility
        self.CAMERA_OFFSET = [x, y]



    def set_camera_resolution(self, width: int, height: int) -> bool:
        """
        Set the camera resolution.
        
        Args:
            width (int): Desired width in pixels
            height (int): Desired height in pixels
            
        Returns:
            bool: True if resolution was set successfully
        """
        # Set width and height
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify the resolution was set
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != width or actual_height != height:
            print(f"Warning: Could not set exact resolution. Actual resolution: {actual_width}x{actual_height}")
            return False
            
        return True

    def get_image_dimensions(self):
        """Get the dimensions of the camera frame."""
        return (self.width, self.height)

    def find_component(self, template_path, angle = 0):
        """
        Find a component in the camera frame and determine its rotation.
        Uses coarse-to-fine search and early termination for better performance.

        Args:
            template_path: Path to template image file
            angle: Expected angle offset (default 0)

        Returns:
            tuple: (center_x, center_y, rotation_angle) if component found
            None: If component not found
        """
        # Load template image with caching for performance
        self.search_frame = self.frame.copy()
        if template_path not in self._template_cache:
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                raise ValueError(f"Could not load template image: {template_path}")
            self._template_cache[template_path] = template
        else:
            template = self._template_cache[template_path]

        image_height, image_width = self.search_frame.shape[:2]
        self.IMAGE_CENTER = (image_width // 2, image_height // 2)

        # Downsample for faster initial search (2x smaller)
        scale_factor = 0.5
        gray = cv2.cvtColor(self.search_frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # Cache downsampled template if not already cached
        template_small_key = f"{template_path}_small"
        if template_small_key not in self._template_cache:
            template_small = cv2.resize(template, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            self._template_cache[template_small_key] = template_small
        else:
            template_small = self._template_cache[template_small_key]
        
        best_match = None
        best_score = -1
        best_angle = 0
        best_template_shape = None
        
        # Stage 1: Coarse search with 5-degree steps on downsampled image
        coarse_angles = list(range(-15+angle, 16+angle, 5))
        coarse_best_angle = 0
        coarse_best_score = -1
        
        for test_angle in coarse_angles:
            # Rotate template
            h, w = template_small.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), test_angle, 1.0)
            
            # Calculate new bounding box size to contain the rotated image
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix for the new size
            matrix[0, 2] += (new_w / 2) - (w / 2)
            matrix[1, 2] += (new_h / 2) - (h / 2)
            
            rotated = cv2.warpAffine(template_small, matrix, (new_w, new_h))
            
            # Template matching on downsampled image
            result = cv2.matchTemplate(gray_small, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > coarse_best_score:
                coarse_best_score = max_val
                coarse_best_angle = test_angle
            
            # Early termination if we found an excellent match
            if max_val > 0.95:
                break
        
        # Stage 2: Fine search with 1-degree steps around best coarse angle on full-res image
        fine_start = coarse_best_angle - 6
        fine_end = coarse_best_angle + 6
        
        for test_angle in range(fine_start, fine_end + 1, 2):
            # Rotate template at full resolution
            h, w = template.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), test_angle, 1.0)
            
            # Calculate new bounding box size
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix
            matrix[0, 2] += (new_w / 2) - (w / 2)
            matrix[1, 2] += (new_h / 2) - (h / 2)
            
            rotated = cv2.warpAffine(template, matrix, (new_w, new_h))
            
            # Template matching
            result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = max_loc
                best_angle = test_angle
                best_template_shape = rotated.shape
            
            # Early termination for excellent matches
            if max_val > 0.95:
                break
        
        if best_score > 0.6:
            # Use the actual rotated template dimensions
            h, w = best_template_shape
            top_left = best_match
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (top_left[0] + w//2, top_left[1] + h//2)
            
            # Draw bounding box and center point
            cv2.rectangle(self.search_frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(self.search_frame, center, 5, (0, 255, 0), -1)
            
            # Draw image center crosshair
            cv2.line(self.search_frame, (self.IMAGE_CENTER[0]-20, self.IMAGE_CENTER[1]), 
                    (self.IMAGE_CENTER[0]+20, self.IMAGE_CENTER[1]), (0, 0, 255), 2)
            cv2.line(self.search_frame, (self.IMAGE_CENTER[0], self.IMAGE_CENTER[1]-20), 
                    (self.IMAGE_CENTER[0], self.IMAGE_CENTER[1]+20), (0, 0, 255), 2)
            
            # Add match quality and rotation text
            cv2.putText(self.search_frame, f"Match: {best_score:.2f}", (50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.search_frame, f"Rotation: {best_angle}°", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.is_component_detected = True
            return (center, best_angle)
        else:
            self.is_component_detected = False
            return (None, None)

    def determine_rotation(self, template_path):
        """
        Determine the rotation of a component in the camera frame.
        #THIS SCRIPT IS INCOMPLETE!
        """
        #Load template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:    
            raise ValueError(f"Could not load template image: {template_path}")
        
        #Get image dimensions
        image_height, image_width = self.search_frame.shape[:2]
        self.IMAGE_CENTER = (image_width // 2, image_height // 2)
        return False
        

    def find_tool_position(self, downsample=True):
        """
        Locate the tool in the camera frame using saved HSV and circle parameters.
        Handles multiple circle detection with user input if needed.
        Optimized for speed with optional downsampling.

        Args:
            downsample: If True, process downsampled images for faster detection (default True)

        Returns:
            tuple: (x, y) pixel coordinates of detected tool center
            None: If no tool can be detected
        """
        # Load parameters from camera-specific file
        params_file = f"vision_params_camera{self.camera_number}_{self.target}.json"
        try:
            with open(params_file, 'r') as f:
                params = json.load(f)
        except FileNotFoundError:
            print(f"No vision parameters found for camera {self.camera_number} and target {self.target}. Run vision_tools.py first to create parameter file.")
            return None

        # Extract parameters
        hsv_lower = np.array(params['hsv_lower'])
        hsv_upper = np.array(params['hsv_upper'])
        circle_params = params['circle_params']

        # Downsampling factor for speed
        scale = 0.5 if downsample else 1.0

        def mouse_callback(event, x, y, flags, circles):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Find which circle was clicked
                for circle in circles[0]:
                    cx, cy = circle[0], circle[1]
                    r = circle[2]
                    # Check if click was inside circle
                    if (x - cx)**2 + (y - cy)**2 <= r**2:
                        mouse_callback.selected_circle = (int(cx / scale), int(cy / scale))
                        print("Circle Selected")
                        return

        frame_count = 0
        while True:
            # Get frame (no buffer clearing in loop for speed)
            frame = self.capture_frame()
            if frame is None:
                continue

            # Downsample for faster processing
            if downsample:
                frame_small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            # Process frame - optimize by skipping masked image creation
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

            # Apply mask and convert to gray in one step (more efficient)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_and(gray, gray, mask=mask)

            # Find circles with scaled parameters
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
                    # Single circle found - scale back to original coordinates
                    print("Single Circle Found")
                    x = int(circles[0][0][0] / scale)
                    y = int(circles[0][0][1] / scale)
                    return (x, y)

                elif len(circles[0]) > 1:
                    # Check distances between circles
                    max_distance = 0
                    for i in range(len(circles[0])):
                        for j in range(i + 1, len(circles[0])):
                            dist = np.sqrt((circles[0][i][0] - circles[0][j][0])**2 +
                                         (circles[0][i][1] - circles[0][j][1])**2)
                            max_distance = max(max_distance, dist)

                    if max_distance > (200 * scale):
                        # Circles are far apart, need user input
                        output = frame_small.copy()

                        # Draw all circles
                        for circle in circles[0]:
                            cv2.circle(output, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                            cv2.circle(output, (circle[0], circle[1]), 2, (0, 0, 255), 3)

                        # Set up mouse callback
                        mouse_callback.selected_circle = None
                        cv2.namedWindow('Select Tool')
                        cv2.setMouseCallback('Select Tool', mouse_callback, circles)

                        # Display instructions
                        cv2.putText(output, "Click the correct circle", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        while mouse_callback.selected_circle is None:
                            cv2.imshow('Select Tool', output)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                cv2.destroyWindow('Select Tool')
                                return None

                        cv2.destroyWindow('Select Tool')
                        return mouse_callback.selected_circle

                    else:
                        # Circles are close, take average and scale back
                        avg_x = int(np.mean([circle[0] for circle in circles[0]]) / scale)
                        avg_y = int(np.mean([circle[1] for circle in circles[0]]) / scale)
                        return (avg_x, avg_y)

            # Display every 3rd frame to reduce overhead
            frame_count += 1
            if frame_count % 3 == 0:
                display_frame = frame_small if downsample else frame
                cv2.imshow('Searching for tool...', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return None

    def set_hsv_thresholds(self, lower: np.array, upper: np.array):
        """
        Set HSV thresholds for tool detection.
        
        Args:
            lower (np.array): Lower bounds for HSV [H, S, V]
            upper (np.array): Upper bounds for HSV [H, S, V]
            
        Raises:
            ValueError: If arrays are not the correct shape or values
        """
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("HSV threshold arrays must have shape (3,)")
            
        self.hsv_lower = lower
        self.hsv_upper = upper

    def initialize_camera(self) -> bool:
        """
        Initialize the camera and check if it's working.
        
        Returns:
            bool: True if camera initialization was successful
        """
        try:
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            self.camera.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
            
    def capture_frame(self, clear_buffer=False) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Args:
            clear_buffer: If True, clear camera buffer before capturing (use sparingly)

        Returns:
            Optional[np.ndarray]: The captured frame or None if capture failed
        """
        if self.camera is None:
            print("Camera not initialized")
            return None

        # Only clear buffer if explicitly requested (e.g., after long pause)
        if clear_buffer:
            for _ in range(3):  # Reduced from 10 to 3
                self.camera.read()

        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None
        self.frame = frame
        return frame
        
    def apply_hsv_threshold(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply HSV thresholding to the frame.

        Args:
            frame: Input frame in BGR format

        Returns:
            np.ndarray: Binary mask after thresholding
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Apply threshold
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Note: Removed morphological operations - 1x1 kernel had no effect
        # Add morphology back with proper kernel size if needed for noise reduction

        return mask
        
    def _circle_key(self, circle: np.ndarray) -> Tuple[int, int, int]:
        """Create a unique key for a circle based on its position and radius."""
        x, y, r = circle
        # Round to nearest 10 pixels to account for small variations
        return (round(x/10)*10, round(y/10)*10, round(r/10)*10)
        
    def _update_circle_history(self, circles: Optional[np.ndarray]) -> List[np.ndarray]:
        """
        Update circle history and return only stable circles.
        
        Args:
            circles: Current frame's detected circles
            
        Returns:
            List[np.ndarray]: List of stable circles
        """
        if circles is None:
            circles = np.array([])
            
        # Convert to list of circles if not empty
        current_circles = circles[0] if len(circles) > 0 else []
        
        # Increment count for circles seen in this frame
        seen_keys = set()
        for circle in current_circles:
            key = self._circle_key(circle)
            seen_keys.add(key)
            self.circle_history[key] += 1
            
        # Decrement count for circles not seen in this frame
        for key in list(self.circle_history.keys()):
            if key not in seen_keys:
                self.circle_history[key] -= 1
                if self.circle_history[key] <= 0:
                    del self.circle_history[key]
                    
        # Return only circles that have been seen for at least min_frames
        stable_circles = []
        for circle in current_circles:
            key = self._circle_key(circle)
            if self.circle_history[key] >= self.min_frames:
                stable_circles.append(circle)
                
        return stable_circles
        
    def detect_circles(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Detect circles in the frame using Hough Circle Transform, only within the HSV mask.
        Optimized to reduce redundant color conversions.

        Args:
            frame: Input frame in BGR format
            mask: Binary mask from HSV thresholding

        Returns:
            Tuple[np.ndarray, np.ndarray, List]: (Output frame with circles drawn, Masked color image with circles, List of detected circles)
        """
        # Convert to grayscale first, then apply mask (more efficient)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        # Note: Removed Gaussian blur with 1x1 kernel (has no effect)
        # Add back with proper kernel size if needed: cv2.GaussianBlur(masked_gray, (5, 5), 0)

        # Detect circles
        circles = cv2.HoughCircles(
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        # Create copies for drawing
        output = frame.copy()

        # Update circle history and get stable circles
        stable_circles = self._update_circle_history(circles)

        # Create masked color image only if we have circles to draw (lazy evaluation)
        if stable_circles:
            masked_color = cv2.bitwise_and(frame, frame, mask=mask)
            masked_output = masked_color.copy()
            print(f"Detected {len(stable_circles)} stable circles")  # Debug info

            for circle in stable_circles:
                x, y, r = circle
                # Draw on original frame
                cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(output, (int(x), int(y)), 2, (0, 0, 255), 3)

                # Draw on masked image
                cv2.circle(masked_output, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(masked_output, (int(x), int(y)), 2, (0, 0, 255), 3)
        else:
            # No circles - create minimal masked output
            masked_color = cv2.bitwise_and(frame, frame, mask=mask)
            masked_output = masked_color
            print("No stable circles detected")  # Debug info

        return output, masked_output, stable_circles
        
    def calibrate_camera(self, 
                        pattern_size: Tuple[int, int] = (9, 6),
                        square_size: float = 0.025) -> bool:
        """
        Calibrate the camera using a chessboard pattern.
        
        Args:
            pattern_size: Number of inner corners in the chessboard pattern (width, height)
            square_size: Size of each square in the chessboard pattern in meters
            
        Returns:
            bool: True if calibration was successful
        """
        # TODO: Implement camera calibration using chessboard pattern
        pass
        
    def transform_coordinates(self, 
                            image_point: Tuple[float, float],
                            z_height: float) -> Tuple[float, float, float]:
        """
        Transform image coordinates to machine coordinates.
        
        Args:
            image_point: (x, y) coordinates in the image
            z_height: Height of the point in machine coordinates
            
        Returns:
            Tuple[float, float, float]: (X, Y, Z) coordinates in machine space
        """
        # TODO: Implement coordinate transformation
        pass
        
    def close(self):
        """Release the camera."""
        if self.camera is not None:
            self.camera.release()

    def __del__(self):
        """Ensure camera is released on deletion."""
        self.close()

    def set_circle_params(self, dp=1, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=50):
        """
        Set parameters for circle detection using HoughCircles.
        
        Args:
            dp (float): Inverse ratio of accumulator resolution to image resolution
            minDist (int): Minimum distance between centers of detected circles
            param1 (int): Upper threshold for edge detection
            param2 (int): Threshold for center detection
            minRadius (int): Minimum radius of detected circles
            maxRadius (int): Maximum radius of detected circles
        """
        self.circle_params = {
            'dp': dp,
            'minDist': minDist,
            'param1': param1,
            'param2': param2,
            'minRadius': minRadius,
            'maxRadius': maxRadius
        }

    def cleanup(self):
        """
        Clean up resources and close windows.
        """
        cv2.destroyAllWindows()
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()
            print("Camera released")

def list_available_cameras(max_cameras: int = 10) -> List[int]:
    """
    List all available cameras by trying to open each one.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List[int]: List of available camera IDs
    """
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Callback functions for trackbars
def on_hue_lower(val):
    global pipeline
    pipeline.hsv_lower[0] = val

def on_sat_lower(val):
    global pipeline
    pipeline.hsv_lower[1] = val

def on_val_lower(val):
    global pipeline
    pipeline.hsv_lower[2] = val

def on_hue_upper(val):
    global pipeline
    pipeline.hsv_upper[0] = val

def on_sat_upper(val):
    global pipeline
    pipeline.hsv_upper[1] = val

def on_val_upper(val):
    global pipeline
    pipeline.hsv_upper[2] = val

def on_dp(val):
    global pipeline
    pipeline.dp = max(1, val)

def on_min_dist(val):
    global pipeline
    pipeline.min_dist = max(1, val)

def on_param1(val):
    global pipeline
    pipeline.param1 = max(1, val)

def on_param2(val):
    global pipeline
    pipeline.param2 = max(1, val)

def on_min_radius(val):
    global pipeline
    pipeline.min_radius = val

def on_max_radius(val):
    global pipeline
    pipeline.max_radius = max(pipeline.min_radius + 1, val)

def on_brightness(val):
    global pipeline
    pipeline.brightness = val
    if pipeline.camera is not None:
        pipeline.camera.set(cv2.CAP_PROP_BRIGHTNESS, val)

def on_contrast(val):
    global pipeline
    pipeline.contrast = val
    if pipeline.camera is not None:
        pipeline.camera.set(cv2.CAP_PROP_CONTRAST, val)





if __name__ == "__main__":
    import json
    import os

    # Get camera number and target from command line or use defaults
    camera_number = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    target = sys.argv[2] if len(sys.argv) > 2 else 'tool'
    
    if target not in ['tool', 'camera','target']:
        print("Invalid target. Must be either 'tool' or 'camera' or 'target'")
        sys.exit(1)
    
    # Default parameter file location (camera and target specific)
    PARAMS_FILE = f"vision_params_camera{camera_number}_{target}.json"

    # Load parameters if file exists
    if os.path.exists(PARAMS_FILE):
        try:
            with open(PARAMS_FILE, 'r') as f:
                params = json.load(f)
                print(f"Loaded parameters from {PARAMS_FILE}")
        except Exception as e:
            print(f"Error loading parameters: {e}")
            params = {
                'hsv_lower': [0, 0, 0],
                'hsv_upper': [180, 255, 255],
                'circle_params': {
                    'dp': 1,
                    'minDist': 20,
                    'param1': 100,
                    'param2': 20,
                    'minRadius': 1,
                    'maxRadius': 20
                }
            }
    else:
        # Default parameters
        params = {
            'hsv_lower': [0, 0, 0],
            'hsv_upper': [180, 255, 255],
            'circle_params': {
                    'dp': 1,
                    'minDist': 20,
                    'param1': 100,
                    'param2': 20,
                    'minRadius': 10,
                    'maxRadius': 20
                }
        }

    # Initialize vision tools with camera number and target
    vision = VisionTools(camera_number, target)

    # Create windows for trackbars
    cv2.namedWindow('HSV Controls')
    cv2.namedWindow('Circle Controls')
    cv2.namedWindow('Camera Feed')

    # Create trackbars for HSV
    cv2.createTrackbar('H min', 'HSV Controls', params['hsv_lower'][0], 180, lambda x: None)
    cv2.createTrackbar('S min', 'HSV Controls', params['hsv_lower'][1], 255, lambda x: None)
    cv2.createTrackbar('V min', 'HSV Controls', params['hsv_lower'][2], 255, lambda x: None)
    cv2.createTrackbar('H max', 'HSV Controls', params['hsv_upper'][0], 180, lambda x: None)
    cv2.createTrackbar('S max', 'HSV Controls', params['hsv_upper'][1], 255, lambda x: None)
    cv2.createTrackbar('V max', 'HSV Controls', params['hsv_upper'][2], 255, lambda x: None)

    # Create trackbars for circle detection
    cv2.createTrackbar('dp', 'Circle Controls', int(params['circle_params']['dp']*10), 10, lambda x: None)  # dp*10 for decimal control
    cv2.createTrackbar('minDist', 'Circle Controls', params['circle_params']['minDist'], 100, lambda x: None)
    cv2.createTrackbar('param1', 'Circle Controls', params['circle_params']['param1'], 200, lambda x: None)
    cv2.createTrackbar('param2', 'Circle Controls', params['circle_params']['param2'], 100, lambda x: None)
    cv2.createTrackbar('minRadius', 'Circle Controls', params['circle_params']['minRadius'], 399, lambda x: None)
    cv2.createTrackbar('maxRadius', 'Circle Controls', params['circle_params']['maxRadius'], 400, lambda x: None)

    print("\nTesting find_tool_position function:")
    print("1. First run the parameter setting mode")
    print("2. Press 'q' to save parameters and start tool detection test")
    
    # First part: Parameter setting mode
    try:
        while True:
            # Get current trackbar values
            hsv_lower = np.array([
                cv2.getTrackbarPos('H min', 'HSV Controls'),
                cv2.getTrackbarPos('S min', 'HSV Controls'),
                cv2.getTrackbarPos('V min', 'HSV Controls')
            ])
            
            hsv_upper = np.array([
                cv2.getTrackbarPos('H max', 'HSV Controls'),
                cv2.getTrackbarPos('S max', 'HSV Controls'),
                cv2.getTrackbarPos('V max', 'HSV Controls')
            ])

            circle_params = {
                'dp': cv2.getTrackbarPos('dp', 'Circle Controls') / 10.0,
                'minDist': cv2.getTrackbarPos('minDist', 'Circle Controls'),
                'param1': cv2.getTrackbarPos('param1', 'Circle Controls'),
                'param2': cv2.getTrackbarPos('param2', 'Circle Controls'),
                'minRadius': cv2.getTrackbarPos('minRadius', 'Circle Controls'),
                'maxRadius': cv2.getTrackbarPos('maxRadius', 'Circle Controls')
            }

            # Get and process frame
            frame = vision.capture_frame()
            if frame is not None:
                # Make a copy for drawing
                output = frame.copy()
                
                # Convert to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Create mask
                mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                
                # Apply mask to original image
                masked = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Find circles using HoughCircles
                gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=circle_params['dp'],
                    minDist=circle_params['minDist'],
                    param1=circle_params['param1'],
                    param2=circle_params['param2'],
                    minRadius=circle_params['minRadius'],
                    maxRadius=circle_params['maxRadius']
                )
                
                # Draw circles if found
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # Draw outer circle
                        cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # Draw center point
                        cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)
                        # Add radius text
                        cv2.putText(gray, f"r={i[2]}", (i[0]+10, i[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Show only the output image
                cv2.imshow('Camera Feed', gray)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Save parameters and break
                params = {
                    'hsv_lower': hsv_lower.tolist(),
                    'hsv_upper': hsv_upper.tolist(),
                    'circle_params': circle_params
                }
                
                with open(PARAMS_FILE, 'w') as f:
                    json.dump(params, f, indent=4)
                print(f"Parameters saved to {PARAMS_FILE}")
                break
    
    finally:
        #continue
        cv2.destroyAllWindows()
    
    # Second part: Test tool detection
    print("\nStarting tool detection test")
    print("- Move the tool in front of the camera")
    print("- If multiple circles are found >200px apart, click the correct one")
    print("- Press 'q' to quit the test")
    
#    vision = VisionTools(
    try:
        while True:
            tool_pos = vision.find_tool_position()
            if tool_pos is not None:
                print(f"Tool found at position: {tool_pos}")
                
                # Get a frame and draw the position
                ret, frame = vision.camera.read()
                if ret and frame is not None:  # Make sure frame is valid
                    # Convert coordinates to integers
                    x, y = map(int, tool_pos)
                    
                    # Create a copy of the frame to draw on
                    output = frame.copy()
                    
                    # Draw crosshair at tool position
                    cv2.line(output, (x-20, y), (x+20, y), (0, 255, 0), 2)
                    cv2.line(output, (x, y-20), (x, y+20), (0, 255, 0), 2)
                    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                    
                    # Add position text
                    cv2.putText(output, f"({x}, {y})", (x+10, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    cv2.imshow('Tool Detection Test', output)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        vision.cleanup()
        print("Test complete") 
