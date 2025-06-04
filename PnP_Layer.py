import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataCollection import DataCollection
from calibration import CalibrateToolheads
from vision_tools import VisionTools
import time

class PnPLayer:
    def __init__(self, config_file: str, calibrate_tool: bool = False):
        """
        Initialize PnP layer with configuration file.
        
        Args:
            config_file (str): Path to JSON configuration file containing component placements
        """
        if calibrate_tool:
            self.printer = CalibrateToolheads()
        self.camera = VisionTools()
        self.load_config(config_file)
        
    def load_config(self, config_file: str) -> None:
        """
        Load component placement configuration from JSON file.
        
        Expected JSON format:
        {
            "components": [
                {
                    "type": "component_name",
                    "template_image": "path/to/template.jpg",
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
            "vacuum_pin": "fan0"
        }
        """
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
                
            # Load template images for each component type
            self.templates = {}
            for component in self.config['components']:
                template = cv2.imread(component['template_image'], cv2.IMREAD_GRAYSCALE)
                if template is None:
                    raise ValueError(f"Could not load template image: {component['template_image']}")
                self.templates[component['type']] = template
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            raise ValueError(f"Error loading configuration: {str(e)}")
            
    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.8) -> Optional[Tuple[float, float, float]]:
        """
        Match template in image and return position and rotation.
        
        Args:
            image: Grayscale image to search in
            template: Template image to match
            threshold: Minimum match quality (0-1)
            
        Returns:
            Tuple of (x, y, rotation_degrees) or None if no match found
        """
        # Convert images to grayscale if they aren't already
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(template.shape) > 2:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
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
            result = cv2.matchTemplate(image, rotated, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score:
                best_score = max_val
                best_match = max_loc
                best_angle = angle
                
        if best_score < threshold:
            return None
            
        # Convert to center coordinates
        center_x = best_match[0] + template.shape[1]/2
        center_y = best_match[1] + template.shape[0]/2
        
        return (center_x, center_y, best_angle)
        
    def control_vacuum(self, state: bool) -> None:
        """
        Control vacuum pump for picking/placing components.
        
        Args:
            state: True to turn on vacuum, False to turn off
        """
        if state:
            self.printer.send_gcode_command(f"M106 P{self.config['vacuum_pin']} S255")
        else:
            self.printer.send_gcode_command(f"M106 P{self.config['vacuum_pin']} S0")
            
    def check_component_alignment(self) -> Optional[Tuple[float, float, float]]:
        """
        Check component alignment using bottom camera.
        
        Returns:
            Tuple of (x_offset, y_offset, rotation) or None if component not found
        """
        # Capture image from bottom camera
        frame = self.camera.capture_frame()
        if frame is None:
            return None
            
        # Try to match current component template
        result = self.match_template(
            frame, 
            self.templates[self.current_component['type']]
        )
        
        if result is None:
            return None
            
        # Calculate offsets from center of image
        image_center = (frame.shape[1]/2, frame.shape[0]/2)
        x_offset = result[0] - image_center[0]
        y_offset = result[1] - image_center[1]
        rotation = result[2]
        
        return (x_offset, y_offset, rotation)
        
    def place_components(self) -> None:
        """
        Main function to place all components according to configuration.
        """
        # Select the correct tool
        self.printer.send_gcode_command(f"T{self.config['tool_number']}")
        
        for component in self.config['components']:
            self.current_component = component
            print(f"\nPlacing component type: {component['type']}")
            
            for placement in component['placements']:
                # Move to reel and pick up component
                print("Moving to reel location...")
                reel = component['reel_location']
                self.printer.send_gcode_command(
                    f"G0 X{reel['x']} Y{reel['y']} Z{reel['z']} F6000"
                )
                time.sleep(0.5)
                self.control_vacuum(True)
                time.sleep(0.5)
                
                # Move to safe Z height
                self.printer.send_gcode_command("G0 Z50 F6000")
                
                # Move to bottom camera for alignment check
                print("Checking component alignment...")
                camera_pos = self.printer.camera_location
                self.printer.send_gcode_command(
                    f"G0 X{camera_pos[0]} Y{camera_pos[1]} Z{camera_pos[2]} F6000"
                )
                time.sleep(0.5)
                
                # Check alignment
                alignment = self.check_component_alignment()
                if alignment is None:
                    print("Warning: Could not detect component, skipping placement")
                    self.control_vacuum(False)
                    continue
                    
                x_offset, y_offset, rot_offset = alignment
                print(f"Detected offsets: X={x_offset:.2f}, Y={y_offset:.2f}, R={rot_offset:.2f}")
                
                # Move to placement location, accounting for offsets
                print("Moving to placement location...")
                self.printer.send_gcode_command("G0 Z50 F6000")  # Safe height first
                
                # Calculate final position with offsets
                final_x = placement['x'] - x_offset
                final_y = placement['y'] - y_offset
                final_r = placement['rotation'] - rot_offset
                
                # Move to position
                self.printer.send_gcode_command(
                    f"G0 X{final_x} Y{final_y} F6000"
                )
                self.printer.send_gcode_command(
                    f"G0 Z{placement['z']} F1200"
                )
                
                # Place component
                time.sleep(0.5)
                self.control_vacuum(False)
                time.sleep(0.5)
                
                # Move back to safe height
                self.printer.send_gcode_command("G0 Z50 F6000")
                
        print("\nComponent placement complete!")
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.printer.close()
        self.camera.cleanup()

if __name__ == "__main__":
    # Example usage
    pnp = PnPLayer("placement_config.json")
    try:
        pnp.place_components()
    finally:
        pnp.cleanup()