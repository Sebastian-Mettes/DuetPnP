import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataCollection import DataCollection
from calibration import CalibrateToolheads
from vision_tools import VisionTools
from feed import Feeder
import time

class PnPLayer:
    def __init__(self, config_file: str, calibrate_tool: bool = False):
        """
        Initialize PnP layer with configuration file.
        
        Args:
            config_file (str): Path to JSON configuration file containing component placements
            calibrate_tool (bool): Whether to run tool calibration on initialization
        """
        self.printer = CalibrateToolheads()
        self.camera_lower = VisionTools(0)  # Upward Facing Camera
        self.camera_upper = VisionTools(2)  # Downward Facing Camera
        self.feeder = Feeder()  # Initialize feeder system
        self.load_config(config_file)
        self.load_camera_offset()
        
        
        
        # Home the feeder system
        self.home_feeder()
        
        if calibrate_tool:
            self.calibrate_tools()
            
    def load_config(self, config_file: str) -> None:
        """
        Load component placement configuration from JSON file.
        
        Expected JSON format:
        {
            "components": [
                {
                    "type": "component_name",
                    "upper_template": "path/to/upper_template.jpg",
                    "lower_template": "path/to/lower_template.jpg",
                    "feed_number": 0,
                    "reel_location": {"x": 0, "y": 0, "z": 0},
                    "placements": [
                        {
                            "x": 0,
                            "y": 0,
                            "z": 0,
                            "rotation": 0
                        }
                    ]
                }
            ],
            "tool_number": 0,
            "vacuum_pin": "fan0",
            "solenoid_pin": "fan1"
        }
        """
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
                
            # Load template images for each component type (upper and lower camera)
            self.upper_templates = {}
            self.lower_templates = {}
            for component in self.config['components']:
                # Load upper camera template (for component identification)
                upper_template = cv2.imread(component['upper_template'], cv2.IMREAD_GRAYSCALE)
                if upper_template is None:
                    print(component)
                    raise ValueError(f"Could not load upper template image: {component['upper_template']}")
                self.upper_templates[component['type']] = upper_template
                
                # Load lower camera template (for alignment and orientation)
                lower_template = cv2.imread(component['lower_template'], cv2.IMREAD_GRAYSCALE)
                if lower_template is None:
                    print(component)
                    raise ValueError(f"Could not load lower template image: {component['lower_template']}")
                self.lower_templates[component['type']] = lower_template
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
            
    def load_camera_offset(self) -> None:
        """Load camera offset from calibration file."""
        try:
            with open('camera_offset.json', 'r') as f:
                self.camera_offset = json.load(f)
            print("Loaded camera offset:", self.camera_offset)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load camera offset: {str(e)}")
            self.camera_offset = {'X': 0, 'Y': 0, 'Z': 0}
            
    def calibrate_tools(self) -> None:
        """Calibrate all tools using vision system."""
        print("Calibrating tools...")
        self.printer.calibrate_with_camera(2)  # Calibrate upper camera (Tool 3)
        self.printer.calibrate_with_camera(1)  # Calibrate picker tool (Tool 2)
        print("Tool calibration complete!")
        
    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.5, auto_center: bool = True,
                      camera_type: str = "upper") -> Optional[Tuple[float, float, float]]:
        """
        Match template in image and return position and rotation.
        Includes automatic centering functionality and visual feedback.
        
        Args:
            image: Grayscale image to search in
            template: Template image to match
            threshold: Minimum match quality (0-1)
            auto_center: Whether to automatically center the target
            camera_type: "upper" or "lower" camera for movement calculations
            
        Returns:
            Tuple of (x, y, rotation_degrees) or None if no match found
        """
        # Convert images to grayscale if they aren't already
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(template.shape) > 2:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions and calculate center
        image_height, image_width = image.shape[:2]
        IMAGE_CENTER = (image_width // 2, image_height // 2)
        
        # Constants for automatic centering
        TOLERANCE = 2  # Pixels from center considered "centered"
        INITIAL_PIXELS_TO_MM = 0.015  # Initial conversion factor
        MAX_ITERATIONS = 20  # Maximum number of centering attempts
        
        # Create window for visual feedback
        window_name = f'Template Matching - {camera_type.capitalize()} Camera'
        cv2.namedWindow(window_name)
        
        iteration = 0
        pixels_to_mm = INITIAL_PIXELS_TO_MM
        
        while iteration < MAX_ITERATIONS:
            # Template matching with rotation
            best_match = None
            best_score = -1
            best_angle = 0
            
            # Try different rotations
            for angle in range(-180, 180, 5):  # 5-degree steps
                # Rotate template
                matrix = cv2.getRotationMatrix2D(
                    (template.shape[1]/2, template.shape[0]/2), 
                    angle, 1.0
                )
                rotated = cv2.warpAffine(
                    template, matrix, 
                    (template.shape[1], template.shape[0])
                )
                
                # Template matching
                result = cv2.matchTemplate(image, rotated, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = max_loc
                    best_angle = angle
            
            if best_score < threshold:
                print(f"Template match score too low: {best_score}")
                cv2.destroyWindow(window_name)
                return None
            
            # Convert to center coordinates
            center_x = best_match[0] + template.shape[1]/2
            center_y = best_match[1] + template.shape[0]/2
            
            # Create visual feedback image
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
            
            # Draw crosshairs at image center
            cv2.line(display_image, (IMAGE_CENTER[0]-20, IMAGE_CENTER[1]), (IMAGE_CENTER[0]+20, IMAGE_CENTER[1]), (0, 0, 255), 2)
            cv2.line(display_image, (IMAGE_CENTER[0], IMAGE_CENTER[1]-20), (IMAGE_CENTER[0], IMAGE_CENTER[1]+20), (0, 0, 255), 2)
            
            # Draw target outline and center
            h, w = template.shape
            top_left = best_match
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (int(center_x), int(center_y))
            
            # Draw rectangle around detected target
            cv2.rectangle(display_image, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(display_image, center, 5, (0, 255, 0), -1)
            
            # Draw line from target center to image center
            cv2.line(display_image, center, IMAGE_CENTER, (255, 0, 0), 2)
            
            # Add text information
            cv2.putText(display_image, f"Match: {best_score:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, f"Angle: {best_angle}°", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, f"Center: ({center_x:.1f}, {center_y:.1f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow(window_name, display_image)
            
            # Check if we're centered within tolerance
            x_offset = IMAGE_CENTER[0] - center_x
            y_offset = IMAGE_CENTER[1] - center_y
            
            if abs(x_offset) <= TOLERANCE and abs(y_offset) <= TOLERANCE:
                print(f"Target successfully centered! Score: {best_score:.3f}, Angle: {best_angle}°")
                cv2.destroyWindow(window_name)
                return (center_x, center_y, best_angle)
            
            # Auto-centering logic
            if auto_center and iteration < MAX_ITERATIONS - 1:
                # Get current machine position
                self.printer.send_gcode_command("M114", check=False)
                current_pos = self.printer.parse_position(self.printer.response)
                
                # Calculate move distance using current conversion factor
                x_move = -y_offset * pixels_to_mm
                y_move = x_offset * pixels_to_mm
                
                # Calculate new position
                new_x = current_pos['X'] + x_move
                new_y = current_pos['Y'] + y_move
                
                # Move to new position slowly
                self.printer.send_gcode_command(f"G0 X{new_x:.3f} Y{new_y:.3f} F1200", check=False)
                print(f"Centering iteration {iteration + 1}: offset (pixels) = ({x_offset:.1f}, {y_offset:.1f}), move (mm) = ({x_move:.3f}, {y_move:.3f})")
                time.sleep(1.5)  # Wait for move to complete and camera image to update
                
                # Update image for next iteration
                if camera_type == "upper":
                    image = self.camera_upper.capture_frame()
                else:
                    image = self.camera_lower.capture_frame()
                
                if image is None:
                    print("Could not capture updated frame")
                    cv2.destroyWindow(window_name)
                    return None
                
                # Convert to grayscale if needed
                if len(image.shape) > 2:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # Manual mode or max iterations reached
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Template matching cancelled by user")
                    cv2.destroyWindow(window_name)
                    return None
                elif key == ord('c'):
                    print("Continuing without centering")
                    cv2.destroyWindow(window_name)
                    return (center_x, center_y, best_angle)
            
            iteration += 1
        
        print("Failed to center target after maximum iterations")
        cv2.destroyWindow(window_name)
        return (center_x, center_y, best_angle)
        
    def control_vacuum(self, state: bool) -> None:
        """
        Control vacuum pump for picking/placing components.
        
        Args:
            state: True to turn on vacuum, False to turn off
        """
        if state:
            self.printer.send_gcode_command(f"M106 P{self.config['vacuum_pin']} S40")
        else:
            self.printer.send_gcode_command(f"M106 P{self.config['vacuum_pin']} S0")
            
    def control_solenoid(self, state: bool) -> None:
        """
        Control solenoid valve for picking/placing components.
        
        Args:
            state: True to open solenoid, False to close solenoid
        """
        if state:
            self.printer.send_gcode_command(f"M106 P{self.config['solenoid_pin']} S2")
        else:
            self.printer.send_gcode_command(f"M106 P{self.config['solenoid_pin']} S0")
            
    def check_component_alignment(self) -> Optional[Tuple[float, float, float]]:
        """
        Check component alignment using lower camera.
        
        Returns:
            Tuple of (x_offset, y_offset, rotation) or None if component not found
        """
        # Capture image from lower camera
        frame = self.camera_lower.capture_frame()
        if frame is None:
            return None
            
        # Try to match current component template
        result = self.match_template(
            frame, 
            self.lower_templates[self.current_component['type']],
            camera_type="lower"
        )
        
        if result is None:
            return None
            
        # Calculate offsets from center of image in pixels
        image_center = (frame.shape[1]/2, frame.shape[0]/2)
        x_offset_pixels = result[0] - image_center[0]
        y_offset_pixels = result[1] - image_center[1]
        
        # Convert pixel offsets to machine coordinates (mm)
        # Using the same conversion factor as in match_template
        pixels_to_mm = 0.015  # Conversion factor from pixels to mm
        # Apply coordinate system transformation (same as in match_template)
        x_offset = -y_offset_pixels * pixels_to_mm
        y_offset = x_offset_pixels * pixels_to_mm
        
        rotation = result[2]
        
        return (x_offset, y_offset, rotation)
        
    def find_component_on_reel(self) -> Optional[Tuple[float, float, float]]:
        """
        Find component on reel using upper camera.
        First moves camera to the reel_location specified in config, then looks for component.
        
        Returns:
            Tuple of (x, y, rotation) or None if component not found
        """
        # Get reel location from config
        if 'reel_location' not in self.current_component:
            raise ValueError(f"Component '{self.current_component['type']}' is missing required 'reel_location' in configuration")
        
        reel_location = self.current_component['reel_location']
        reel_x = reel_location['x']
        reel_y = reel_location['y']
        reel_z = reel_location.get('z', 150)  # Default to safe Z height if not specified
        
        print(f"Moving camera to reel location: X={reel_x}, Y={reel_y}, Z={reel_z}")
        
        # Move camera to reel location
        self.printer.send_gcode_command("T3")  # Select upper camera tool
        self.printer.send_gcode_command("M106 P3 S255")  # Turn on upper camera LED
        time.sleep(0.5)
        
        # Move to safe height first, then to reel location
        self.printer.send_gcode_command("G90")
        time.sleep(1.0)
        self.printer.send_gcode_command("G0 X0 Y0 F6000")
        time.sleep(1.0)
        self.printer.send_gcode_command("G0 Z150 F6000")
        time.sleep(1.0)
        self.printer.send_gcode_command(f"G0 X{reel_x} Y{reel_y} F6000")
        time.sleep(1.0)
        self.printer.send_gcode_command(f"G0 Z{reel_z} F6000")
        time.sleep(1.5)  # Wait for movement to complete and camera image to stabilize
        
        # Capture image from upper camera
        frame = self.camera_upper.capture_frame()
        if frame is None:
            print("Could not capture frame from upper camera")
            self.printer.send_gcode_command("M106 P3 S0")  # Turn off LED
            return None
            
        # Try to match current component template
        result = self.match_template(
            frame, 
            self.upper_templates[self.current_component['type']],
            camera_type="upper"
        )
        
        if result is None:
            print("Could not find component in camera view")
            self.printer.send_gcode_command("M106 P3 S0")  # Turn off LED
            return None
            
        # Convert pixel coordinates to machine coordinates and apply camera offset
        pixels_to_mm = 0.015  # Conversion factor from pixels to mm
        # Apply coordinate system transformation (same as in match_template)
        x = -result[1] * pixels_to_mm + self.camera_offset['X']
        y = result[0] * pixels_to_mm + self.camera_offset['Y']
        rotation = result[2]
        
        print(f"Found component at machine coordinates: X={x:.3f}, Y={y:.3f}, rotation={rotation:.1f}°")
        
        # Turn off LED
        self.printer.send_gcode_command("M106 P3 S0")
        
        return (x, y, rotation)
        
    def place_components(self) -> None:
        """
        Main function to place all components according to configuration.
        """
        # Move to safe height with upper camera (Tool 3)
        self.printer.send_gcode_command("T3")  # Select upper camera tool
        self.printer.send_gcode_command("G0 Z150 F6000")  # Move to safe height
        self.feeder = Feeder()
        self.feeder.home()
        for component in self.config['components']:
            self.current_component = component
            component_type = component['type']
            feed_number = component.get('feed_number', 0)  # Default to feed 0 if not specified
            print(f"\nPlacing component type: {component_type} from feed {feed_number}")
            
            for placement in component['placements']:
                # Retry loop for component placement
                max_retries = 3
                retry_count = 0
                placement_successful = False
                
                while retry_count < max_retries and not placement_successful:
                    try:
                        # Feed component using the feeder system
                        print(f"Feeding component from feed {feed_number}...")
                        try:
                            self.feeder.feed(feed_number)
                            print(f"Successfully fed component from feed {feed_number}")
                        except Exception as e:
                            print(f"Warning: Failed to feed component from feed {feed_number}: {str(e)}")
                            retry_count += 1
                            continue
                        
                        # Find component on reel using upper camera
                        print("Locating component on reel...")
                        component_pos = self.find_component_on_reel()
                        if component_pos is None:
                            print("Warning: Could not locate component on reel, skipping")
                            retry_count += 1
                            continue
                            
                        # Switch to picker tool (Tool 2)
                        print("Switching to picker tool...")
                        self.printer.send_gcode_command("T2")
                        
                        # Move to component location and pick up
                        print("Picking up component...")
                        self.printer.send_gcode_command("G0 Z150 F6000")  # Safe height first
                        
                        # Move to component position (no Y offset calculation needed)
                        self.printer.send_gcode_command(
                            f"G0 X{component_pos[0]} Y{component_pos[1]} F6000"
                        )
                        time.sleep(1.0)
                        
                        # Move down to pick up component
                        self.printer.send_gcode_command("G0 Z50 F6000")
                        self.control_vacuum(True)
                        
                        time.sleep(0.5)
                        self.printer.send_gcode_command("G0 Z-8 F6000")
                        self.control_solenoid(True)
                        time.sleep(0.5)
                        self.printer.send_gcode_command("G0 Z150 F6000")  # Back to safe height
                        
                        # Move to lower camera for alignment check
                        print("Checking component alignment...")
                        camera_pos = self.printer.camera_location
                        self.printer.send_gcode_command(
                            f"G0 X{camera_pos[0]} Y{camera_pos[1]} Z{camera_pos[2]} F6000"
                        )
                        time.sleep(2.5)
                        
                        # Check alignment
                        alignment = self.check_component_alignment()
                        if alignment is None:
                            print("Warning: Could not detect component, skipping placement")
                            self.control_vacuum(False)
                            retry_count += 1
                            continue
                            
                        x_offset, y_offset, detected_rotation = alignment
                        target_rotation = placement['rotation']
                        rot_offset = target_rotation - detected_rotation
                        
                        print(f"Detected rotation: {detected_rotation:.2f}°, Target rotation: {target_rotation:.2f}°")
                        print(f"Detected offsets: X={x_offset:.2f}, Y={y_offset:.2f}, Rotation error: {rot_offset:.2f}°")
                        
                        # Reorient tool if needed
                        if abs(rot_offset) > 1.0:  # If rotation error > 1 degree
                            print(f"Reorienting tool by {rot_offset:.2f} degrees...")
                            # Rotate C axis by the opposite of the rotation error
                            rotation_command = f"G0 C{rot_offset:.2f} F600"
                            self.printer.send_gcode_command(rotation_command, check=False)
                            time.sleep(1.0)  # Wait for rotation to complete
                            print(f"Tool rotated by {rot_offset:.2f} degrees")
                        print("Checking component position... 2nd time")
                        camera_pos = self.printer.camera_location
                        self.printer.send_gcode_command(
                            f"G0 X{camera_pos[0]} Y{camera_pos[1]} Z{camera_pos[2]} F6000"
                        )
                        time.sleep(2.5)
                        
                        # Check alignment
                        alignment = self.check_component_alignment()
                        # Note: Removed second rotation check to avoid issues
                        x_offset, y_offset, detected_rotation = alignment
                        # Move to placement location, accounting for offsets
                        print("Moving to placement location...")
                        self.printer.send_gcode_command("G0 Z150 F6000")  # Safe height first
                        
                        # Calculate final position with offsets and camera offset
                        final_x = placement['x'] - x_offset 
                        final_y = placement['y'] - y_offset
                        
                        # Move to position
                        self.printer.send_gcode_command(
                            f"G0 X{final_x} Y{final_y} F6000"
                        )
                        self.printer.send_gcode_command(
                            f"G0 Z{placement['z']} F1200"
                        )
                        
                        # Place component
                        time.sleep(0.5)
                        self.control_solenoid(False)
                        self.control_vacuum(False)
                        time.sleep(0.5)
                        
                        # Move back to safe height
                        self.printer.send_gcode_command("G0 Z150 F6000")
                        
                        # Verification step: Show upper camera view of placed component
                        print("Verifying component placement...")
                        self.verify_component_placement(placement, final_x, final_y)
                        
                        # If we get here, placement was successful and user chose to continue
                        placement_successful = True
                        
                    except Exception as e:
                        if "User requested retry" in str(e):
                            retry_count += 1
                            print(f"Retrying placement (attempt {retry_count}/{max_retries})...")
                            # Turn off vacuum and solenoid in case they were left on
                            self.control_vacuum(False)
                            self.control_solenoid(False)
                            continue
                        elif "User requested quit" in str(e):
                            print("User requested to quit placement process.")
                            return  # Exit the entire placement process
                        else:
                            print(f"Error during placement: {str(e)}")
                            retry_count += 1
                            if retry_count >= max_retries:
                                print(f"Failed to place component after {max_retries} attempts, moving to next component.")
                                break
                            continue
                
                if not placement_successful:
                    print(f"Component placement failed after {max_retries} attempts, moving to next component.")
        
        print("\nComponent placement complete!")
        
    def verify_component_placement(self, placement: dict, final_x: float, final_y: float) -> None:
        """
        Verify component placement using upper camera.
        Shows camera view of placed component and allows user to continue or retry.
        
        Args:
            placement: Placement configuration for the component
            final_x: Final X coordinate where component was placed
            final_y: Final Y coordinate where component was placed
        """
        # Switch to upper camera tool (Tool 3)
        self.printer.send_gcode_command("T3")
        
        # Turn on upper camera ring light
        self.printer.send_gcode_command("M106 P3 S255")  # Turn on upper camera ring light (Fan 3)
        time.sleep(0.5)
        
        # Move to safe height first, then to placement location
        self.printer.send_gcode_command("G0 Z150 F6000")
        time.sleep(1.0)
        self.printer.send_gcode_command(f"G0 X{final_x} Y{final_y} F6000")
        time.sleep(1.0)
        self.printer.send_gcode_command("G0 Z166 F6000")  # Move to camera height
        time.sleep(1.5)  # Wait for movement to complete and camera image to stabilize
        
        # Create window for verification
        cv2.namedWindow('Component Verification')
        
        print("Component verification:")
        print("- Press 'C' to continue to next component")
        print("- Press 'R' to retry this placement")
        print("- Press 'Q' to quit")
        
        while True:
            # Capture frame from upper camera
            frame = self.camera_upper.capture_frame()
            if frame is None:
                continue
                
            # Add text overlay with instructions
            cv2.putText(frame, "Component Verification", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'C' to continue, 'R' to retry, 'Q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Position: X={final_x:.2f}, Y={final_y:.2f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Component: {self.current_component['type']}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Component Verification', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c') or key == ord('C'):
                print("Continuing to next component...")
                break
            elif key == ord('r') or key == ord('R'):
                print("Retrying component placement...")
                # Turn off camera and close window
                self.printer.send_gcode_command("M106 P3 S0")  # Turn off upper camera ring light
                cv2.destroyWindow('Component Verification')
                # Raise an exception to trigger retry (will be caught in place_components)
                raise Exception("User requested retry")
            elif key == ord('q') or key == ord('Q'):
                print("Quitting component placement...")
                # Turn off camera and close window
                self.printer.send_gcode_command("M106 P3 S0")  # Turn off upper camera ring light
                cv2.destroyWindow('Component Verification')
                # Raise an exception to stop placement
                raise Exception("User requested quit")
        
        # Turn off upper camera ring light
        self.printer.send_gcode_command("M106 P3 S0")  # Turn off upper camera ring light
        time.sleep(0.5)
        
        # Deselect camera tool before closing window
        self.printer.send_gcode_command("T-1")
        cv2.destroyWindow('Component Verification')
        
        print("Component verification complete!")
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.printer.close()
        self.camera_lower.cleanup()
        self.camera_upper.cleanup()
        # Feeder cleanup is handled automatically when the object goes out of scope

    def home_feeder(self) -> None:
        """Home the feeder system to establish reference position."""
        print("Homing feeder system...")
        try:
            self.feeder.home()
            print("Feeder homing complete!")
        except Exception as e:
            print(f"Warning: Feeder homing failed: {str(e)}")
            print("Continuing without feeder homing...")

if __name__ == "__main__":
    # Example usage
    pnp = PnPLayer("placement_config.json", calibrate_tool=False) #Already Calibrated!
    try:
        pnp.place_components()
    finally:
        pnp.cleanup()