import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from calibration import CalibrateToolheads as Printer
from vision_tools import VisionTools as Camera
from feed import Feeder
import time





class PnP:
    def __init__(self):
        #First setup a single viewscreen for all actions that require a camera.
        self.create_display_window()

        #Then initialize all the cameras.
        self.initialize_cameras()

        #Then initialize the printer.
        self.printer = Printer()

        #Then initialize the feeder.
        self.feeder = Feeder()
        self.feeder.home()

        #Then initialize the Instructions JSON file.
        self.load_config("placement_config.json")

    def create_display_window(self, window_name: str = 'PnP Camera View'):
        """
        Creates a single OpenCV window for displaying camera images.
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_FULLSCREEN)

        
    def display_image(self, image: np.ndarray, window_name: str = 'PnP Camera View', text: str = None):
        """
        Updates the display window with a new image and optional text overlay.
        
        Args:
            window_name: Name of window to update
            image: Image array to display
            text: Optional text to overlay on image
        """
        # Create a copy of the image to avoid modifying original
        display_image = image.copy()
        
        # Add text overlay if provided
        if text:
            cv2.putText(display_image, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
        # Show the image
        cv2.imshow(window_name, display_image)
        cv2.waitKey(1)  # Required to update display


    def initialize_cameras(self):
        """
        Initializes all cameras for vision processing.
        """
        self.camera_lower = Camera(0)   #Upward Facing Camera for finding toolheads
        self.camera_upper = Camera(2)   #Downward Facing Camera for finding feeder parts
        self.camera_lower.set_fixed_camera_offset(0, 0) #Manually determined values for camera offset
        self.camera_upper.set_fixed_camera_offset(0.85, 0.40) #Manually determined values for camera offset
    
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

    def move_camera(self, x: float, y: float, z: float):
        """
        Moves the camera to a specified position and checks no movements happen if camera is below z = 100
        """
        #Get current position
        current_pos = self.printer.get_current_position()


        #Check current tool is tool 3 (the camera tool):
        current_tool = self.printer.get_current_tool()
        if current_tool != 3:
            #Switch to tool 3
            self.printer.send_gcode_command(f"T3")
            time.sleep(3.5) #Sufficient delay for tool change.

        #Check if camera is below z = 100
        if current_pos[2] < 100:
            #Move camera to z = 100 
            self.printer.send_gcode_command(f"G0 Z150 F6000") #Manually determined value for z = 150 for camera focus.
        
        #Move to new position
        self.printer.send_gcode_command(f"G0 X{x} Y{y} Z{z} F6000")


        

    def run(self):
        
        #Loop through all components in the config file:
        for component in self.config['components']:
            #for each placement in the component:
            for placement in component['placements']:

                #Move to pickup location
                self.move_camera(component['reel_location']['x'], component['reel_location']['y'], component['reel_location']['z'])
                #While looking for component, continuously update the display window with the camera image.
                
                while not self.camera_upper.is_component_detected():
                    self.image = self.camera_upper.take_picture()
                    self.display_image(self.image, "PnP Camera View", "Looking for component")

                    #Check if component is detected:
                    if self.camera_upper.is_component_detected():
                        #Move to placement location:
                        self.move_camera(placement['x'], placement['y'], placement['z'])
                        #Take picture of component:

        #First move camera to part location in feeder (from config file):
        self.move_camera(self.config['components'][0]['reel_location']['x'], self.config['components'][0]['reel_location']['y'], self.config['components'][0]['reel_location']['z'])

        #Then run the PnP process.




















if __name__ == "__main__":
    pnp = PnP()
    pnp.run()