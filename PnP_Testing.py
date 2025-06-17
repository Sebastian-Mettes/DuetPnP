import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from PnP_Layer import PnPLayer
from calibration import CalibrateToolheads
from vision_tools import VisionTools
import time
import os

class PnPTesting:
    def __init__(self):
        """Initialize PnP testing system."""
        self.printer = CalibrateToolheads()
        self.camera_upper = VisionTools(2)  # Downward Facing Camera
        self.camera_lower = VisionTools(0)  # Upward Facing Camera
        self.load_camera_offset()
        
    def load_camera_offset(self) -> None:
        """Load camera offset from calibration file."""
        try:
            with open('camera_offset.json', 'r') as f:
                self.camera_offset = json.load(f)
            print("Loaded camera offset:", self.camera_offset)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load camera offset: {str(e)}")
            self.camera_offset = {'X': 0, 'Y': 0, 'Z': 0}
            
    def find_target_with_camera(self, location: Tuple[float, float, float], template_path: str) -> None:
        """
        Move camera to position and look for target using template matching.
        Includes automatic centering functionality.
        
        Args:
            location: (X, Y, Z) coordinates to move camera to
            template_path: Path to template image file
        """
        # Load template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Could not load template image: {template_path}")
            
        # Select camera tool (Tool 3)
        self.printer.send_gcode_command("T3")
        
        # Turn on upper camera ring light
        self.printer.send_gcode_command("M106 P3 S255")  # Turn on upper camera ring light (Fan 3)
        time.sleep(0.5)
        
        # Move to initial position
        x, y, z = location
        self.printer.send_gcode_command(f"G0 Z150 F6000") #Move up to avoid collision with the tool
        time.sleep(1.5) 
        self.printer.send_gcode_command(f"G0 X{x} Y{y} Z{z}F6000")
        time.sleep(1.5)  # Wait for movement to complete
        
        # Create window for camera view
        cv2.namedWindow('Target Detection')
        
        # Constants for automatic centering
        TOLERANCE = 2  # Pixels from center considered "centered"
        INITIAL_PIXELS_TO_MM = 0.015  # Initial conversion factor
        MAX_ITERATIONS = 20  # Maximum number of centering attempts
        
        # Movement step size for manual control
        STEP_SIZE = 0.25
        
        # Get image dimensions and calculate center
        frame = self.camera_upper.capture_frame()
        if frame is None:
            raise RuntimeError("Could not capture initial frame")
        image_height, image_width = frame.shape[:2]
        IMAGE_CENTER = (image_width // 2, image_height // 2)
        
        # Automatic centering mode
        auto_center = False
        iteration = 0
        pixels_to_mm = INITIAL_PIXELS_TO_MM
        
        while True:
            # Capture frame
            frame = self.camera_upper.capture_frame()
            if frame is None:
                continue
                
            # Convert frame to grayscale for template matching
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Draw rectangle around best match
            h, w = template.shape
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (top_left[0] + w//2, top_left[1] + h//2)
            
            # Draw rectangle and center point
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
            # Draw image center crosshair
            cv2.line(frame, (IMAGE_CENTER[0]-20, IMAGE_CENTER[1]), (IMAGE_CENTER[0]+20, IMAGE_CENTER[1]), (0, 0, 255), 2)
            cv2.line(frame, (IMAGE_CENTER[0], IMAGE_CENTER[1]-20), (IMAGE_CENTER[0], IMAGE_CENTER[1]+20), (0, 0, 255), 2)
            
            # Add match quality text
            cv2.putText(frame, f"Match: {max_val:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add mode text
            mode_text = "Auto-Centering" if auto_center else "Manual Control"
            cv2.putText(frame, f"Mode: {mode_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Target Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):  # Toggle auto-centering
                auto_center = not auto_center
                iteration = 0
                print(f"Switched to {'auto-centering' if auto_center else 'manual'} mode")
            elif not auto_center:  # Manual control only when not auto-centering
                if key == ord('w'):  # Move +Y
                    self.printer.send_gcode_command(f"G91 G0 Y{STEP_SIZE} F1200 G90")
                elif key == ord('s'):  # Move -Y
                    self.printer.send_gcode_command(f"G91 G0 Y-{STEP_SIZE} F1200 G90")
                elif key == ord('a'):  # Move -X
                    self.printer.send_gcode_command(f"G91 G0 X-{STEP_SIZE} F1200 G90")
                elif key == ord('d'):  # Move +X
                    self.printer.send_gcode_command(f"G91 G0 X{STEP_SIZE} F1200 G90")
                elif key == ord('r'):  # Move +Z
                    self.printer.send_gcode_command(f"G91 G0 Z{STEP_SIZE} F1200 G90")
                elif key == ord('f'):  # Move -Z
                    self.printer.send_gcode_command(f"G91 G0 Z-{STEP_SIZE} F1200 G90")
                time.sleep(0.01)  # Small delay to prevent too rapid movement
            
            # Auto-centering logic
            if auto_center and max_val > 0.5:  # Only attempt centering if we have a good match
                x_offset = IMAGE_CENTER[0] - center[0]
                y_offset = IMAGE_CENTER[1] - center[1]
                
                # Check if we're centered within tolerance
                if abs(x_offset) <= TOLERANCE and abs(y_offset) <= TOLERANCE:
                    print("Target successfully centered!")
                    # Get current position and save it
                    self.printer.send_gcode_command("M114")
                    current_pos = self.printer.parse_position(self.printer.response)
                    
                    # Save target location to JSON file
                    target_location = {
                        'x': current_pos['X'],
                        'y': current_pos['Y'],
                        'z': current_pos['Z'],
                        'template_path': template_path
                    }
                    
                    try:
                        # Get absolute path for the file
                        file_path = os.path.abspath('target_location.json')
                        print(f"Attempting to save to: {file_path}")
                        
                        # Check if directory is writable
                        directory = os.path.dirname(file_path)
                        if not os.access(directory, os.W_OK):
                            print(f"Warning: No write permission in directory: {directory}")
                        
                        # Try to create/overwrite the file
                        with open(file_path, 'w') as f:
                            json.dump(target_location, f, indent=4)
                            f.flush()  # Ensure data is written to disk
                            os.fsync(f.fileno())  # Force system to write to disk
                            
                        # Verify the file was written
                        if os.path.exists(file_path):
                            print(f"Target location saved successfully to: {file_path}")
                            print(f"Target location: X={current_pos['X']:.3f}, Y={current_pos['Y']:.3f}, Z={current_pos['Z']:.3f}")
                        else:
                            print("Error: File was not created")
                            
                    except PermissionError as e:
                        print(f"Permission error saving target location: {str(e)}")
                    except IOError as e:
                        print(f"IO error saving target location: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error saving target location: {str(e)}")
                        import traceback
                        print("Full error traceback:")
                        print(traceback.format_exc())
                    
                    auto_center = False
                    continue
                
                if iteration >= MAX_ITERATIONS:
                    print("Failed to center after maximum iterations")
                    auto_center = False
                    continue
                
                # Get current machine position
                self.printer.send_gcode_command("M114")
                current_pos = self.printer.parse_position(self.printer.response)
                
                # Calculate move distance using current conversion factor
                x_move = -x_offset * pixels_to_mm
                y_move = y_offset * pixels_to_mm
                
                # Calculate new position
                new_x = current_pos['X'] + x_move
                new_y = current_pos['Y'] + y_move
                
                # Print debug information
                print(f"Current position: X={current_pos['X']:.3f}, Y={current_pos['Y']:.3f}")
                print(f"Offsets: X={x_offset:.1f}, Y={y_offset:.1f} pixels")
                print(f"Move distances: X={x_move:.3f}, Y={y_move:.3f} mm")
                print(f"New position: X={new_x:.3f}, Y={new_y:.3f}")
                
                # Move to new position slowly
                self.printer.send_gcode_command(f"G0 X{new_x:.3f} Y{new_y:.3f} F1200")
                print(f"Centering iteration {iteration}: offset (pixels) = ({x_offset}, {y_offset})")
                time.sleep(1.5)  # Wait for move to complete and camera image to update
                
                iteration += 1
            
        # Turn off upper camera ring light
        self.printer.send_gcode_command("M106 P3 S0")  # Turn off upper camera ring light
        time.sleep(0.5)
        
        # Deselect camera tool before closing window
        self.printer.send_gcode_command("T-1")
        cv2.destroyWindow('Target Detection')
        
    def pickup_and_verify(self) -> bool:
        """
        Pick up target using Tool 2 and verify with lower camera.
        Uses saved target position from target_location.json and accounts for camera offsets.
        
        Returns:
            bool: True if pickup and verification successful
        """
        # Load saved target position
        try:
            with open('target_location.json', 'r') as f:
                target_location = json.load(f)
            print(f"Loaded target location: X={target_location['x']:.3f}, Y={target_location['y']:.3f}, Z={target_location['z']:.3f}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading target location: {str(e)}")
            return False
            
        # Load camera offsets
        try:
            with open('camera_offset.json', 'r') as f:
                camera_offset = json.load(f)
            print(f"Loaded camera offset: X={camera_offset['X']:.3f}, Y={camera_offset['Y']:.3f}, Z={camera_offset['Z']:.3f}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load camera offset: {str(e)}")
            camera_offset = {'X': 0, 'Y': 0, 'Z': 0}
            
        # Switch to picker tool (Tool 2)
        self.printer.send_gcode_command("T2")
        
        # First move to safe Z height
        self.printer.send_gcode_command("G0 Z150 F6000")
        time.sleep(1.5)
        
        # Turn on vacuum before moving
        self.printer.send_gcode_command("M106 P1 S102")  # Turn on vacuum pump (Fan 1)
        time.sleep(0.5)

        # Calculate target position with camera offset
        target_x = target_location['x'] - camera_offset['X']
        target_y = target_location['y'] - camera_offset['Y']
        target_z = target_location['z']
        
        print(f"Moving to adjusted target position: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}")

        # Move to target position
        self.printer.send_gcode_command(f"G0 X{target_x} Y{target_y} F6000")
        self.printer.send_gcode_command(f"G0 Z10 F6000")
        self.printer.send_gcode_command(f"G0 Z-7.6 F300")
        time.sleep(2.5)
        self.printer.send_gcode_command("M106 P2 S15")  # Open solenoid valve (Fan 2)

        time.sleep(3.5) #Long Delay for observation
        
        # Move up to safe height
        self.printer.send_gcode_command("G0 Z150 F6000")
        time.sleep(1.5)
        
        # Move to camera position
        camera_pos = self.printer.camera_location
        self.printer.send_gcode_command(f"G0 X{camera_pos[0]} Y{camera_pos[1]} Z{camera_pos[2]} F6000")
        time.sleep(1.5)
        
        # Turn on lower camera ring light
        self.printer.send_gcode_command("M106 P4 S255")  # Turn on lower camera ring light (Fan 4)
        time.sleep(0.5)
        
        # Create window for verification
        cv2.namedWindow('Pickup Verification')
        
        while True:
            # Capture frame from lower camera
            frame = self.camera_lower.capture_frame()
            if frame is None:
                continue
                
            # Show frame
            cv2.imshow('Pickup Verification', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Turn off lower camera ring light
                self.printer.send_gcode_command("M106 P4 S0")  # Turn off lower camera ring light
                time.sleep(0.5)
                cv2.destroyWindow('Pickup Verification')
                return True
            elif key == ord('c'):
                # Turn off vacuum and lower camera ring light
                self.printer.send_gcode_command("M106 P1 S0")  # Turn off vacuum pump
                self.printer.send_gcode_command("M106 P2 S0")  # Close solenoid valve
                self.printer.send_gcode_command("M106 P4 S0")  # Turn off lower camera ring light
                time.sleep(0.5)
                cv2.destroyWindow('Pickup Verification')
                return False
                
    def determine_rotation(self, template_path: str) -> None:
        """
        Determine rotation of picked up object using template matching.
        
        Args:
            template_path: Path to template image file
        """
        # Load template image
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise ValueError(f"Could not load template image: {template_path}")
            
        # Turn on lower camera ring light
        self.printer.send_gcode_command("M106 P4 S255")  # Turn on lower camera ring light (Fan 4)
        time.sleep(0.5)
            
        # Create window for rotation detection
        cv2.namedWindow('Rotation Detection')
        
        while True:
            # Capture frame from lower camera
            frame = self.camera_lower.capture_frame()
            if frame is None:
                continue
                
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            best_match = None
            best_score = -1
            best_angle = 0
            
            # Try different rotations
            for angle in range(0, 360, 5):  # 5-degree steps
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
                result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = max_loc
                    best_angle = angle
            
            # Draw results
            if best_match is not None:
                h, w = template.shape
                top_left = best_match
                bottom_right = (top_left[0] + w, top_left[1] + h)
                center = (top_left[0] + w//2, top_left[1] + h//2)
                
                # Draw rectangle and center point
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                
                # Add angle and match quality text
                cv2.putText(frame, f"Angle: {best_angle}°", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Match: {best_score:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Rotation Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        # Turn off lower camera ring light
        self.printer.send_gcode_command("M106 P4 S0")  # Turn off lower camera ring light
        time.sleep(0.5)
        cv2.destroyWindow('Rotation Detection')
        
    def cleanup(self):
        """Clean up resources."""
        self.printer.close()
        self.camera_upper.cleanup()
        self.camera_lower.cleanup()

if __name__ == "__main__":
    # Example usage
    pnp_test = PnPTesting()
    try:
        # Step 1: Find target with camera
        pnp_test.find_target_with_camera(
            location=(-42.8, 238.1, 144.00),  # Example camera position
            template_path="Resistor_G_Samp.png"  # Path to your template image
        )
        
        # Step 2: Pickup and verify
        if pnp_test.pickup_and_verify():
            # Step 3: Determine rotation
            pnp_test.determine_rotation(template_path="template.png")
    finally:
        pnp_test.cleanup()
