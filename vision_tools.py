import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import defaultdict
import json
import os

class VisionTools:
    """
    A class to handle computer vision operations for toolhead calibration.
    
    This class provides methods to capture and process camera images,
    detect tools, and perform calibration-related vision tasks.
    
    Attributes:
        camera: OpenCV video capture object
        image_width (int): Width of camera frame in pixels
        image_height (int): Height of camera frame in pixels
        hsv_lower (np.array): Lower bounds for HSV color filtering
        hsv_upper (np.array): Upper bounds for HSV color filtering
    """

    def __init__(self):
        """
        Initialize VisionTools with camera and default parameters.
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Get camera resolution
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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

    def get_image_dimensions(self):
        """Get the dimensions of the camera frame."""
        return (self.width, self.height)

    def find_tool_position(self):
        """
        Locate the tool in the camera frame using saved HSV and circle parameters.
        Handles multiple circle detection with user input if needed.
        
        Returns:
            tuple: (x, y) pixel coordinates of detected tool center
            None: If no tool can be detected
        """
        # Load parameters from file
        try:
            with open("vision_params.json", 'r') as f:
                params = json.load(f)
        except FileNotFoundError:
            print("No vision parameters found. Run vision_tools.py first to create parameter file.")
            return None
        
        # Extract parameters
        hsv_lower = np.array(params['hsv_lower'])
        hsv_upper = np.array(params['hsv_upper'])
        circle_params = params['circle_params']
        
        def mouse_callback(event, x, y, flags, circles):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Find which circle was clicked
                for circle in circles[0]:
                    cx, cy = circle[0], circle[1]
                    r = circle[2]
                    # Check if click was inside circle
                    if (x - cx)**2 + (y - cy)**2 <= r**2:
                        mouse_callback.selected_circle = (cx, cy)
                        return
        
        while True:
            # Get frame
            ret, frame = self.camera.read()  # Using correct cv2.VideoCapture method
            if not ret:
                continue
            
            # Process frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
            
            # Find circles
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
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                if len(circles[0]) == 1:
                    # Single circle found
                    return (int(circles[0][0][0]), int(circles[0][0][1]))
                
                elif len(circles[0]) > 1:
                    # Check distances between circles
                    max_distance = 0
                    for i in range(len(circles[0])):
                        for j in range(i + 1, len(circles[0])):
                            dist = np.sqrt((circles[0][i][0] - circles[0][j][0])**2 + 
                                         (circles[0][i][1] - circles[0][j][1])**2)
                            max_distance = max(max_distance, dist)
                    
                    if max_distance > 200:
                        # Circles are far apart, need user input
                        output = frame.copy()
                        
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
                        # Circles are close, take average
                        avg_x = int(np.mean([circle[0] for circle in circles[0]]))
                        avg_y = int(np.mean([circle[1] for circle in circles[0]]))
                        return (avg_x, avg_y)
            
            # No circles found, keep running
            cv2.imshow('Searching for tool...', frame)
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
            
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Optional[np.ndarray]: The captured frame or None if capture failed
        """
        if self.camera is None:
            print("Camera not initialized")
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None
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
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((1,1), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
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
        
        Args:
            frame: Input frame in BGR format
            mask: Binary mask from HSV thresholding
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List]: (Output frame with circles drawn, Masked color image with circles, List of detected circles)
        """
        # Create a masked version of the original color image
        masked_color = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert the masked color image to grayscale for circle detection
        masked_gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(masked_gray, (1, 1), 0)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
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
        masked_output = masked_color.copy()
        
        # Update circle history and get stable circles
        stable_circles = self._update_circle_history(circles)
        
        if stable_circles:
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

    # Default parameter file location
    PARAMS_FILE = "vision_params.json"

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
                    'minRadius': 1,
                    'maxRadius': 20
                }
        }

    # Initialize vision tools
    vision = VisionTools()

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
    cv2.createTrackbar('minRadius', 'Circle Controls', params['circle_params']['minRadius'], 100, lambda x: None)
    cv2.createTrackbar('maxRadius', 'Circle Controls', params['circle_params']['maxRadius'], 200, lambda x: None)

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
                        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # Draw center point
                        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
                        # Add radius text
                        cv2.putText(output, f"r={i[2]}", (i[0]+10, i[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Show only the output image
                cv2.imshow('Camera Feed', output)

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
        cv2.destroyAllWindows()
    
    # Second part: Test tool detection
    print("\nStarting tool detection test")
    print("- Move the tool in front of the camera")
    print("- If multiple circles are found >200px apart, click the correct one")
    print("- Press 'q' to quit the test")
    
    vision = VisionTools()
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