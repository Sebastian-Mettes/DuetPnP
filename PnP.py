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

        #Modifiable Parameters:
        self.TOLERANCE = 2  # Pixels from center considered "centered"
        self.INITIAL_PIXELS_TO_MM = 0.015  # Initial conversion factor
        self.MAX_ITERATIONS = 20  # Maximum number of centering attempts

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
        self.camera_lower = Camera(0, target='tool')   #Upward Facing Camera for finding toolheads
        self.camera_upper = Camera(2, target='tool')   #Downward Facing Camera for finding feeder parts
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

    def move_camera(self, x: float = None, y: float = None, z: float = None):
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
        if current_pos['Z'] < 100:
            #Move camera to z = 100 
            self.printer.linear_move(z=144) #Manually determined value for z = 150 for camera focus.
        
        #Move to new position
        self.printer.linear_move(x,y,z)



    def run(self):        
        #Loop through all components in the config file:
        for component in self.config['components']:
            #for each placement in the component:
            for placement in component['placements']:
                pickup_location = component['reel_location']
                #Move to pickup location
                self.move_camera(component['reel_location']['x'], component['reel_location']['y'], component['reel_focus'])
                #While looking for component, continuously update the display window with the camera image.
                self.printer.send_gcode_command("M106 P3 S255")  # Turn on upper camera ring light (Fan 3)

                while not self.camera_upper.is_component_detected:
                    self.display_image(self.camera_upper.capture_frame(), "PnP Camera View", "Looking for component") #Display the camera image in the window.
                    self.camera_upper.find_component(component['upper_template'])
                #While component is detected, continuously update the display window with the camera image. Centeer component in image, save location:
                centered = False
                while centered == False:
                    self.camera_upper.capture_frame()                   
                    center = None
                    center, rotation = self.camera_upper.find_component(component['upper_template'])
                    self.display_image(self.camera_upper.search_frame, "PnP Camera View", "Centering Component") #Display the camera image in the window.
                    self.printer.send_gcode_command("M114")
                    current_pos = self.printer.parse_position(self.printer.response)
                    if center is not None:
                        if abs(center[0] - self.camera_upper.IMAGE_CENTER[0]) <= 2 and abs(center[1] - self.camera_upper.IMAGE_CENTER[1]) <= 2:
                            centered = True
                            self.display_image(self.camera_upper.search_frame, "PnP Camera View", "Component Centered")
                            new_x = current_pos['X'] 
                            new_y = current_pos['Y']


                        else:
                            x_offset = self.camera_upper.IMAGE_CENTER[0] - center[0]
                            y_offset = self.camera_upper.IMAGE_CENTER[1] - center[1]
                            
                            x_move = -x_offset * self.INITIAL_PIXELS_TO_MM
                            y_move = y_offset * self.INITIAL_PIXELS_TO_MM
                            new_x = current_pos['X'] + x_move 
                            new_y = current_pos['Y'] + y_move 
                            self.printer.send_gcode_command(f"G0 X{new_x:.3f} Y{new_y:.3f} F6000")
                            time.sleep(0.5)
                    
                        #Save the pickup location:
                        pickup_location['x'] = new_x - self.camera_upper.CAMERA_OFFSET[0]
                        pickup_location['y'] = new_y - self.camera_upper.CAMERA_OFFSET[1]
                    else:
                        print("Component lost")
                
                self.printer.send_gcode_command('T2')#Switch to PnP Tool.
                time.sleep(3.5) #Sufficient delay for tool change.

                #Move to placement XY location:
                self.printer.linear_move(x=pickup_location['x'], y=pickup_location['y'])
                
                 #Turn on vacuum: 
                self.printer.send_gcode_command(f"M106 P{self.config['vacuum_pin'][3]} S40") #Turn on vacuum
                
                #Move to Z height for pickup:
                self.printer.linear_move(z=component['reel_location']['z'])
                time.sleep(0.5)  
                                #Open solenoid:
                self.printer.send_gcode_command(f"M106 P{self.config['solenoid_pin'][3]} S6") #Turn on solenoid, value 6 to ensure low current as needed.
                time.sleep(0.5)

                #Move back to Z height of 150:
                self.printer.linear_move(z=150)

                #Move to printer.camera_location:
                self.printer.linear_move(x=self.printer.camera_location[0], y=self.printer.camera_location[1], z=self.printer.camera_location[2])
                time.sleep(2.5)
                oriented = False
                centered = False
                rotated = False
                #turn on lower camera ring light:
                self.printer.send_gcode_command("M106 P4 S255")  # Turn on lower camera ring light (Fan 4)
                time.sleep(0.5)

                while oriented == False:
                    self.camera_lower.capture_frame()
                    #Push image to display window:                    #Find component in image:
                    center = None
                    center,rotation = self.camera_lower.find_component(component['lower_template'])
                    self.display_image(self.camera_lower.search_frame, "PnP Camera View", "Centering Component") #Display the camera image in the window.
                    self.printer.send_gcode_command("M114")
                    current_pos = self.printer.parse_position(self.printer.response)
                    if center is not None and centered is not True:   
                        if abs(center[0] - self.camera_lower.IMAGE_CENTER[0]) > 2 and abs(center[1] - self.camera_lower.IMAGE_CENTER[1]) > 2:
                            x_offset = self.camera_lower.IMAGE_CENTER[0] - center[0] #pixel offset from center
                            y_offset = self.camera_lower.IMAGE_CENTER[1] - center[1]
                            
                            x_move = -x_offset * self.INITIAL_PIXELS_TO_MM
                            y_move = y_offset * self.INITIAL_PIXELS_TO_MM
                            new_x = current_pos['X'] + x_move - self.camera_lower.CAMERA_OFFSET[0]
                            new_y = current_pos['Y'] + y_move - self.camera_lower.CAMERA_OFFSET[1]
                            self.printer.send_gcode_command(f"G0 X{new_x:.3f} Y{new_y:.3f} F6000")
                            time.sleep(0.25)

                        else:                            
                            offset_x = current_pos['X'] - self.camera_lower.CAMERA_OFFSET[0] - self.printer.camera_location[0]
                            offset_y = current_pos['Y'] - self.camera_lower.CAMERA_OFFSET[1] - self.printer.camera_location[1]
                            if rotated is True:
                                centered = True
                                oriented = True
                            else:
                                #send command which will rotate component to desired rotation:
                                self.printer.send_gcode_command(f"G0 C{rotation + placement['rotation']} F6000")
                                time.sleep(0.25)
                                rotated = True
                    




                
                #Move to placement location (X,Y):
                self.printer.linear_move(x=placement['x']+offset_x, y=placement['y']+offset_y)
                #Move to Z height for placement:
                self.printer.linear_move(z=placement['z'])
                
            
                #Turn off solenoid:
                self.printer.send_gcode_command(f"M106 P{self.config['solenoid_pin'][3]} S0") #Turn off solenoid
                time.sleep(0.5)

                #Move back to Z 150:
                self.printer.linear_move(z=150)
                #Turn off vacuum:
                self.printer.send_gcode_command(f"M106 P{self.config['vacuum_pin'][3]} S0") #Turn off vacuum

                #Now use T3 (camera) to verify placement by taking a photo and saving it in a folder (/verification_photos)
                self.printer.send_gcode_command('T3')
                time.sleep(3.5)

                #Turn on camera ring light:
                self.printer.send_gcode_command("M106 P3 S255")  # Turn on upper camera ring light (Fan 3)

                #Move to placement location (X,Y):
                self.printer.linear_move(x=placement['x']+offset_x, y=placement['y']+offset_y)
                time.sleep(1.5)

                #Take photo:
                self.camera_upper.capture_frame()
                self.display_image(self.camera_upper.search_frame, "PnP Camera View", "Verifying Placement")
                cv2.imwrite(f"verification_photos/{component['type']}_{placement['x']}_{placement['y']}_{placement['z']}.png", self.camera_upper.search_frame)

                    #Press C to continue to next component: 
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('c'):
                        break


















if __name__ == "__main__":
    pnp = PnP()
    pnp.run()