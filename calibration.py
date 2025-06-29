from dsf.connections import CommandConnection
from vision_tools import VisionTools
import time
import json
import cv2

VALID_GCODES = [
    # Motion G-codes
    'G0', 'G1', 'G2', 'G3', 'G4', 'G28',
    
    # Probe G-codes
    'G29', 'G30', 'G31', 'G32',
    'G38.2', 'G38.3', 'G38.4', 'G38.5',
    
    # Units and Positioning G-codes
    'G17', 'G18', 'G19', 'G20', 'G21',
    'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59',
    'G59.1', 'G59.2', 'G59.3', 'G60', 'G68', 'G69',
    'G90', 'G91', 'G92',
    
    # Retraction G-codes
    'G10', 'G11',
    
    # M-codes for Machine Control
    'M0', 'M1', 'M3', 'M4', 'M5',
    'M17', 'M18', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25',
    'M26', 'M27', 'M28', 'M29', 'M30', 'M32', 'M36', 'M37',
    'M38', 'M39', 'M42',
    
    # Temperature and fan Control
    'M104', 'M105', 'M106', 'M107', 'M108', 'M109',
    'M116', 'M140', 'M141', 'M143', 'M144', 'M190', 'M191',
    
    # Configuration and Status
    'M111', 'M114', 'M115', 'M119', 'M122',
    'M280', 'M290', 'M291', 'M292', 'M300',
    
    # Settings and Storage
    'M500', 'M501', 'M502', 'M503', 'M505',
    
    # Network and Communication
    'M550', 'M551', 'M552', 'M553', 'M554',
    'M555', 'M556', 'M557', 'M558', 'M559', 'M560',
    
    # Tool Control
    'M563', 'M567', 'M568', 'M569',
    'M569.1', 'M569.2', 'M569.3', 'M569.4', 'M569.5', 'M569.6', 'M569.7',
    
    # Motion Control
    'M201', 'M201.1', 'M203', 'M204', 'M205', 'M206', 'M207', 'M208',
    'M220', 'M221', 'M566', 'M567', 'M568', 'M569', 'M570',
    
    # Special Functions
    'M581', 'M582', 'M584', 'M585', 'M586', 'M587', 'M588', 'M589',
    'M591', 'M592', 'M593', 'M594', 'M595',
    
    # Power and Emergency
    'M80', 'M81', 'M112', 'M999',
    
    # Filament Control
    'M600', 'M701', 'M702', 'M703',

    # Toolhead G-codes
    'T-1', 'T0', 'T1', 'T2', 'T3',
]

class CalibrateToolheads:
    """
    A class to handle calibration of toolheads using machine vision on a Duet-based 3D printer.
    
    This class provides methods to control and calibrate multiple toolheads
    using G-code commands and vision-based calibration.
    """

    def __init__(self):
        """
        Initialize the CalibrateToolheads class.
        
        Sets up the vision tools and establishes a connection to the printer.
        """

        self.connection = CommandConnection(debug=False)
        self.connection.connect()
        self.camera_location = [21.6,-59.7,166.41] #Camera coordinates, XYZ



    def send_gcode_command(self, gcode_command, check=True):
        """
        Send a G-code command to the printer.
        
        Args:
            gcode_command (str): The G-code command to send
            check (bool, optional): Whether to validate the command. Defaults to True.
            
        Raises:
            ValueError: If check=True and command is invalid
        """
        if check:
            # Validate gcode_command format
            if not isinstance(gcode_command, str):
                raise ValueError("G-code command must be a string")
            
            # Extract the base command (without parameters)
            base_command = gcode_command.split()[0]
            
            # Check if it's a valid command
            if base_command not in VALID_GCODES:
                raise ValueError(f"Invalid G-code command: {base_command}")
        
        # Execute the command and save the response
        self.response = self.connection.perform_simple_code(gcode_command)

    def print_response(self):
        """Print the response from the last G-code command."""
        print(self.response)


    def close(self):
        """Close the connection to the printer."""
        self.connection.close()

    def home(self):
        """
        Home all axes of the printer.
        
        Deselects all tools and performs a full homing operation.
        """
        self.send_gcode_command("T-1", check=False)
        self.send_gcode_command("G28", check=False)
        self.print_response()

    def parse_position(self, response):
        """
        Parse M114 response to get current position.
        
        Args:
            response (str): Response string from M114 command
            
        Returns:
            dict: Dictionary containing X, Y, Z positions and optional U, V, W, A axes
            
        Raises:
            ValueError: If response is empty or missing required axes
            ValueError: If position values cannot be converted to float
        """
        if not response:
            raise ValueError("Empty response from M114 command")
        
        position = {}
        valid_axes = ['X', 'Y', 'Z', 'U', 'V', 'W', 'A']  # All possible axes
        required_axes = ['X', 'Y', 'Z']  # Must have these
        
        try:
            for part in response.split():
                if ':' in part:
                    axis, value = part.split(':')
                    if axis in valid_axes:
                        try:
                            position[axis] = float(value)
                        except ValueError:
                            raise ValueError(f"Invalid position value for {axis}: {value}")
        except Exception as e:
            raise ValueError(f"Error parsing M114 response: {str(e)}")
        
        # Check if all required axes are present
        missing_axes = set(required_axes) - set(position.keys())
        if missing_axes:
            raise ValueError(f"Missing required axes in M114 response: {missing_axes}")
        
        # Log any additional axes found
        additional_axes = set(position.keys()) - set(required_axes)
        if additional_axes:
            print(f"Additional axes found: {additional_axes}")
        
        return position

    def parse_tool_offsets(self, response, toolhead_number):
        """
        Parse G10 response to get tool offsets.
        Example response: "Tool 0: offsets X1.300 Y-102.200 Z-18.250 U0.000 V0.000, active/standby temperature(s) 0.0/0.0"
        
        Args:
            response (str): Response string from G10 command
            toolhead_number (int): Tool number to find offsets for
            
        Returns:
            dict: Dictionary containing X, Y, Z offsets and optional U, V axes
            
        Raises:
            ValueError: If response is empty
            ValueError: If tool number not found in response
            ValueError: If offset values cannot be converted to float
        """
        if not response:
            raise ValueError("Empty response from G10 command")
        
        offsets = {'X': 0, 'Y': 0, 'Z': 0}  # Default to zero offsets
        valid_axes = ['X', 'Y', 'Z', 'U', 'V', 'W', 'A']  # All possible axes
        
        try:
            # Check if this is the correct tool response
            if not response.startswith(f"Tool {toolhead_number}:"):
                raise ValueError(f"Response does not match Tool {toolhead_number}")
            
            # Extract the offsets section
            offset_section = response.split("offsets ")[1].split(",")[0]
            
            # Parse each axis offset
            for axis in valid_axes:
                # Look for axis in the response
                axis_start = offset_section.find(axis)
                if axis_start != -1:
                    # Extract value until next space or end
                    value_str = ""
                    i = axis_start + 1
                    while i < len(offset_section) and (offset_section[i].isdigit() or offset_section[i] in '.-'): #Check if the character is a digit or a decimal point or negative sign
                        value_str += offset_section[i]
                        i += 1
                    try:
                        offsets[axis] = float(value_str)
                    except ValueError:
                        raise ValueError(f"Invalid offset value for {axis}: {value_str}")
                    
        except Exception as e:
            raise ValueError(f"Error parsing G10 response: {str(e)}")
        print(f"Offsets: {offsets}")
        return offsets

    def parse_axis_mapping(self, response):
        """
        Parse M563 response to get tool axis mappings.
        Example response: "Tool 0 - drives: 0; heaters (active/standby temps): 2 (0.0/0.0); xmap: X; ymap: Y; zmap: Z; fans: 0; no spindle; status: standby"
        
        Args:
            response (str): Response string from M563 command
            
        Returns:
            dict: Dictionary containing axis mappings (e.g., {'X': 'X', 'Y': 'Y', 'Z': 'Z'} or {'X': 'U', 'Y': 'V', 'Z': 'Z'})
            
        Raises:
            ValueError: If response is empty or invalid format
        """
        if not response:
            raise ValueError("Empty response from M563 command")
        
        mappings = {}
        try:
            # Look for xmap, ymap, zmap in the response
            for mapping in ['xmap:', 'ymap:', 'zmap:']:
                map_start = response.find(mapping)
                if map_start != -1:
                    # Get the character after "xmap: " etc
                    axis_pos = map_start + len(mapping)
                    while axis_pos < len(response) and response[axis_pos].isspace():
                        axis_pos += 1
                    if axis_pos < len(response):
                        source_axis = mapping[0].upper()  # X, Y, or Z
                        mapped_axis = response[axis_pos]  # What it maps to (X, Y, Z, U, V, etc)
                        mappings[source_axis] = mapped_axis
                        
        except Exception as e:
            raise ValueError(f"Error parsing M563 response: {str(e)}")
        
        # Verify we got all required mappings
        required_maps = ['X', 'Y', 'Z']
        missing_maps = set(required_maps) - set(mappings.keys())
        if missing_maps:
            raise ValueError(f"Missing axis mappings in M563 response: {missing_maps}")
        
        return mappings

    def get_current_tool(self):
        """
        Get the current toolhead number using Duet3 object model.
        """
        response = self.send_gcode_command("M409 K\"state.currentTool\"", check=False)
        # Parse the JSON response
        try:
            data = json.loads(response)
            return data.get('result', 0)  # Default to tool 0 if not found
        except json.JSONDecodeError:
            return 0



    def set_tool_offset(self, expected_position, toolhead_number):
        """
        Set the tool offset for a specific toolhead based on expected position.
        Align toolhead over camera, then modify G10 offsets to set current position to expected position.
        Takes into account axis mappings for the tool.
        
        Args:
            expected_position (list): The expected XYZ coordinates [X, Y]
            toolhead_number (int): The number of the toolhead to set the offset for
            
        Raises:
            ValueError: If position or offset parsing fails
            ValueError: If toolhead_number is invalid
            ValueError: If expected_position is invalid
        """
        # Validate toolhead number
        if not isinstance(toolhead_number, int) or toolhead_number < 0:
            raise ValueError("Invalid toolhead number")
        
        # Validate expected_position
        if not isinstance(expected_position, (list, tuple)) or len(expected_position) != 3:
            raise ValueError("Expected position must be a list or tuple of 3 coordinates [X, Y, Z]")
        try:
            expected_position = [float(val) for val in expected_position]
        except (ValueError, TypeError):
            raise ValueError("Expected position coordinates must be numeric values")
        
        try:
            # Get axis mappings for this tool
            self.send_gcode_command(f"M563 P{toolhead_number}", check=False)
            axis_maps = self.parse_axis_mapping(self.response)
            print(f"Axis maps: {axis_maps}")
            # Get current position
            self.send_gcode_command("M114", check=False)
            current_pos = self.parse_position(self.response)
            
            # Get current tool offsets
            self.send_gcode_command(f"G10 P{toolhead_number}", check=False)
            current_offsets = self.parse_tool_offsets(self.response, toolhead_number)
            
            # Calculate new offsets using mapped axes
            new_offsets = {}
            for source_axis, mapped_axis in axis_maps.items():
                # Get the index for X, Y, or Z (0, 1, or 2)
                axis_index = 'XYZ'.index(source_axis)
                print(f"Axis index: {axis_index}")
                # Calculate the difference in the source axis (X, Y, or Z)
                print(f"Current position: {current_pos}")
                print(f"Source axis: {source_axis}")
                print(f"Current position: {current_pos[source_axis]}")
                print(f"Expected position: {expected_position[axis_index]}")
                source_diff = current_pos[source_axis] - expected_position[axis_index]
                print(f"Source difference: {source_diff}")
                # Add this difference to the current offset of the mapped axis
                new_offsets[mapped_axis] = current_offsets[mapped_axis] - source_diff
                print(f"New offsets: {new_offsets}")
            
            # Build G10 command with mapped axes
            offset_params = ' '.join(
                f"{axis}{value:.3f}" for axis, value in new_offsets.items()
            )
            print('offset params built')
            self.send_gcode_command(
                f"G10 P{toolhead_number} {offset_params}",
                check=False
            )
            
            # Print the changes with axis mapping information
            print(f"Axis mappings: {axis_maps}")
            print(f"Current position: X{current_pos['X']:.3f} Y{current_pos['Y']:.3f} Z{current_pos['Z']:.3f}")
            print(f"Expected position: X{expected_position[0]:.3f} Y{expected_position[1]:.3f} Z{expected_position[2]:.3f}")
            print(f"Old offsets: {' '.join(f'{k}{v:.3f}' for k, v in current_offsets.items())}")
            print(f"New offsets: {' '.join(f'{k}{v:.3f}' for k, v in new_offsets.items())}")
            
        except ValueError as e:
            print(f"Error setting tool offset: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error setting tool offset: {str(e)}")
            raise
    
    def linear_move(self, x: float = None, y: float = None, z: float = None):
        """
        Move the tool to a specified position.
        Only moves axes that are provided.
        """
        move_cmd = "G0"
        if x is not None:
            move_cmd += f" X{x}"
        if y is not None:
            move_cmd += f" Y{y}" 
        if z is not None:
            move_cmd += f" Z{z}"
        move_cmd += " F6000"
        self.send_gcode_command(move_cmd, check=False)
        time.sleep(1.5)

    def calibrate_tool_with_camera(self, toolhead_number, target='tool', camera=0, deselect_tool=True):
        """
        Prepare a specific toolhead for calibration and center it in the camera view.
        Uses computer vision to locate the tool and iteratively moves it to the center.
        Once centered, sets the tool offset relative to the camera position.
        
        Args:
            toolhead_number (int): The number of the toolhead to calibrate
            target (str): What to look for - 'tool' or 'camera'
            camera (int): Camera number to use (0 for lower camera, 2 for upper camera)
            deselect_tool (bool): Whether to deselect the tool after calibration
            
        Returns:
            bool: True if calibration successful, False if tool cannot be centered
        """
        # Select the tool and move to initial position
        self.send_gcode_command("T-1", check=False)
        self.send_gcode_command(f"T{toolhead_number}", check=False)
        
        # Move to safe Z height first
        self.send_gcode_command("G0 Z166.41 F6000", check=False)
        
        # Move to approximate camera XY position
        if camera == 0:
            self.send_gcode_command("M106 P4 S255") #Turn on LED Ring for lower camera    
        elif camera == 2:
            self.send_gcode_command("M106 P3 S255") #Turn on LED Ring for upper camera
        else:
            raise ValueError(f"Invalid camera number: {camera}. Must be 0 (lower) or 2 (upper)")
            
        self.send_gcode_command(f"G0 X{self.camera_location[0]} Y{self.camera_location[1]} Z{self.camera_location[2]} F6000", check=False)
        print(f"G0 X{self.camera_location[0]} Y{self.camera_location[1]} Z{self.camera_location[2]} F6000")
        time.sleep(1.5)

        # Initialize vision tools
        vision = VisionTools(camera, target)
        
        # Get image dimensions and calculate center
        image_width, image_height = vision.get_image_dimensions()
        IMAGE_CENTER = (image_width // 2, image_height // 2)
        print(f"Image center: {IMAGE_CENTER}")

        # Mouse callback for target confirmation
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_callback.confirmed = True
                print("Target confirmed by user click")

        # Create window and set mouse callback
        cv2.namedWindow('Confirm Target')
        mouse_callback.confirmed = False
        cv2.setMouseCallback('Confirm Target', mouse_callback)

        print("\nPlease confirm the target:")
        print("1. Look for the detected target in the window")
        print("2. Click on the target to confirm it's correct")
        print("3. Press 'q' to cancel")

        # Wait for target confirmation
        while not mouse_callback.confirmed:
            frame = vision.capture_frame()
            if frame is not None:
                # Get tool position in camera image
                tool_pos = vision.find_tool_position()
                if tool_pos is not None:
                    x_pixel, y_pixel = tool_pos
                    # Draw larger circle at detected position
                    cv2.circle(frame, (x_pixel, y_pixel), 20, (0, 255, 0), 3)  # Increased radius and thickness
                    cv2.putText(frame, "Click to confirm target", (10, 60),  # Moved text down
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Increased font size and thickness
                
                cv2.imshow('Confirm Target', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Target confirmation cancelled")
                cv2.destroyWindow('Confirm Target')
                if camera == 0:
                    self.send_gcode_command("M106 P4 S0")
                elif camera == 2:
                    self.send_gcode_command("M106 P3 S0")
                if deselect_tool:
                    self.send_gcode_command("T-1", check=False)
                return False

        cv2.destroyWindow('Confirm Target')
        print("Target confirmed, proceeding with calibration...")

        # Constants for the centering algorithm
        MAX_ITERATIONS = 20  # Maximum number of attempts to center
        TOLERANCE = 2  # Pixels from center considered "centered"
        INITIAL_PIXELS_TO_MM = 0.015  # Initial conversion factor
        
        iteration = 0
        pixels_to_mm = INITIAL_PIXELS_TO_MM  # Start with initial conversion factor
        previous_pos = None
        previous_pixel_pos = None
        
        while iteration < MAX_ITERATIONS:
            # Get tool position in camera image
            tool_pos = vision.find_tool_position()
            print(f"Tool pixel position: {tool_pos}")
            if tool_pos is None:
                print("Could not detect tool in camera image")
                continue
            
            x_pixel, y_pixel = tool_pos
            x_offset = IMAGE_CENTER[0] - x_pixel
            y_offset = IMAGE_CENTER[1] - y_pixel
            
            # Check if we're centered within tolerance
            if abs(x_offset) <= TOLERANCE and abs(y_offset) <= TOLERANCE:
                print("Tool successfully centered in camera view")
                print(f"Camera location: {self.camera_location}")
                # Set tool offset relative to camera location
                try:
                    self.set_tool_offset(self.camera_location, toolhead_number)
                    print("Tool offset successfully set relative to camera position")
                    if camera == 0:
                        self.send_gcode_command("M106 P4 S0") #Turn off LED Ring for lower camera
                    elif camera == 2:
                        self.send_gcode_command("M106 P3 S0") #Turn off LED Ring for upper camera
                    if deselect_tool:
                        self.send_gcode_command("T-1", check=False)
                    return True
                except Exception as e:
                    print(f"Failed to set tool offset: {str(e)}")
                    if camera == 0:
                        self.send_gcode_command("M106 P4 S0") #Turn off LED Ring for lower camera
                    elif camera == 2:
                        self.send_gcode_command("M106 P3 S0") #Turn off LED Ring for upper camera
                    if deselect_tool:
                        self.send_gcode_command("T-1", check=False)
                    return False
            
            # Get current machine position
            self.send_gcode_command("M114", check=False)
            current_pos = self.parse_position(self.response)
            
            # Store current positions for next iteration
            previous_pos = current_pos
            previous_pixel_pos = (x_pixel, y_pixel)
            
            # Calculate move distance using current conversion factor
            x_move = -y_offset * pixels_to_mm
            y_move = x_offset * pixels_to_mm
            
            # Calculate new position
            new_x = current_pos['X'] + x_move
            new_y = current_pos['Y'] + y_move
            
            # Move to new position slowly
            self.send_gcode_command(f"G0 X{new_x:.3f} Y{new_y:.3f} F1200", check=False)
            print(f"Moved to new position: X{new_x:.3f} Y{new_y:.3f}")
            # Small delay to ensure move is complete and camera image is updated
            time.sleep(1.5)
            
            iteration += 1
            print(f"Centering iteration {iteration}: offset (pixels) = ({x_offset}, {y_offset}), move (mm) = ({x_move:.3f}, {y_move:.3f})")
        
        print("Failed to center tool after maximum iterations")
        if camera == 0:
            self.send_gcode_command("M106 P4 S0") #Turn off LED Ring for lower camera    
        elif camera == 2:
            self.send_gcode_command("M106 P3 S0") #Turn off LED Ring for upper camera
        if deselect_tool:
            self.send_gcode_command("T-1", check=False) #Deselect tool
        return False

    def calibrate_camera_with_camera(self, tool, camera=2):
        """
        Calibrate the upper camera (camera 2) using the lower camera (camera 0) as reference.
        
        Args:
            tool (int): Tool number to use (should be tool 3 - upper camera tool)
            camera (int): Camera number to use (default 2 for upper camera)
            
        Returns:
            bool: True if calibration successful, False otherwise
        """
        # First, use lower camera to position the upper camera tool
        print("Step 1: Positioning upper camera tool using lower camera...")
        if not self.calibrate_tool_with_camera(tool, camera=0, target='camera', deselect_tool=False):
            print("Failed to position upper camera tool using lower camera")
            return False
            
        # Get the position where the tool is centered in lower (calibraiton) camera view
        self.send_gcode_command("M114", check=False)
        lower_camera_position = self.parse_position(self.response)
        print(f"Position when centered in lower camera: {lower_camera_position}")
        
        # Now use upper camera to find the lower camera
        print("\nStep 2: Using upper camera to find lower camera...")
        # Turn on LED for upper camera
        self.send_gcode_command("M106 P3 S255") #Turn on LED Ring for upper camera
        time.sleep(1.5)  # Give LED time to stabilize
        
        # Initialize upper camera with correct target type
        upper_camera = VisionTools(camera, target='camera')  # Use camera 2 to look for camera
        
        # Get image dimensions and calculate center
        image_width, image_height = upper_camera.get_image_dimensions()
        IMAGE_CENTER = (image_width // 2, image_height // 2)
        print(f"Image center: {IMAGE_CENTER}")

        # Mouse callback for target confirmation
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_callback.confirmed = True
                print("Target confirmed by user click")

        # Create window and set mouse callback
        cv2.namedWindow('Confirm Target')
        mouse_callback.confirmed = False
        cv2.setMouseCallback('Confirm Target', mouse_callback)

        print("\nPlease confirm the target:")
        print("1. Look for the detected target in the window")
        print("2. Click on the target to confirm it's correct")
        print("3. Press 'q' to cancel")

        # Wait for target confirmation
        while not mouse_callback.confirmed:
            frame = upper_camera.capture_frame()
            if frame is not None:
                # Get tool position in camera image
                tool_pos = upper_camera.find_tool_position()
                if tool_pos is not None:
                    x_pixel, y_pixel = tool_pos
                    # Draw larger circle at detected position
                    cv2.circle(frame, (x_pixel, y_pixel), 20, (0, 255, 0), 3)  # Increased radius and thickness
                    cv2.putText(frame, "Click to confirm target", (10, 60),  # Moved text down
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)  # Increased font size and thickness
                
                cv2.imshow('Confirm Target', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Target confirmation cancelled")
                cv2.destroyWindow('Confirm Target')
                self.send_gcode_command("M106 P3 S0")
                self.send_gcode_command("T-1", check=False)
                return False

        cv2.destroyWindow('Confirm Target')
        print("Target confirmed, proceeding with calibration...")

        # Constants for the centering algorithm
        MAX_ITERATIONS = 20
        TOLERANCE = 2
        INITIAL_PIXELS_TO_MM = 0.015
        
        iteration = 0
        pixels_to_mm = INITIAL_PIXELS_TO_MM
        previous_pos = None
        previous_pixel_pos = None
        
        while iteration < MAX_ITERATIONS:
            # Get lower camera position in upper camera image
            tool_pos = upper_camera.find_tool_position() #Find the circle identifying the camera lens of lower camera.
            print(f"Lower camera pixel position: {tool_pos}")
            if tool_pos is None:
                print("Could not detect lower camera in upper camera view")
                continue
            
            x_pixel, y_pixel = tool_pos
            x_offset = IMAGE_CENTER[0] - x_pixel
            y_offset = IMAGE_CENTER[1] - y_pixel
            
            # Check if we're centered within tolerance
            if abs(x_offset) <= TOLERANCE and abs(y_offset) <= TOLERANCE:
                print("Lower camera successfully centered in upper camera view")
                # Get final position
                self.send_gcode_command("M114", check=False)
                upper_camera_position = self.parse_position(self.response)
                print(f"Position when centered in upper camera: {upper_camera_position}")
                
                # Calculate the offset between the two positions
                camera_offset = {
                    'X': upper_camera_position['X'] - lower_camera_position['X'],
                    'Y': upper_camera_position['Y'] - lower_camera_position['Y'],
                    'Z': upper_camera_position['Z'] - lower_camera_position['Z']
                }
                print(f"Camera offset: {camera_offset}")
                
                # Save the offset to a file for future use
                try:
                    with open('camera_offset.json', 'w') as f:
                        json.dump(camera_offset, f, indent=4)
                    print("Camera offset saved to camera_offset.json")
                except Exception as e:
                    print(f"Failed to save camera offset: {str(e)}")
                
                # Clean up
                self.send_gcode_command("M106 P3 S0") #Turn off LED Ring for upper camera
                self.send_gcode_command("T-1", check=False)
                return True
            
            # Get current machine position
            self.send_gcode_command("M114", check=False)
            current_pos = self.parse_position(self.response)
            
            # Store current positions for next iteration
            previous_pos = current_pos
            previous_pixel_pos = (x_pixel, y_pixel)
            
            # Calculate move distance using current conversion factor
            x_move = -x_offset * pixels_to_mm
            y_move = y_offset * pixels_to_mm
            
            # Calculate new position
            new_x = current_pos['X'] + x_move
            new_y = current_pos['Y'] + y_move
            
            # Move to new position slowly
            self.send_gcode_command(f"G0 X{new_x:.3f} Y{new_y:.3f} F1200", check=False)
            print(f"Moved to new position: X{new_x:.3f} Y{new_y:.3f}")
            time.sleep(0.5)
            
            iteration += 1
            print(f"Centering iteration {iteration}: offset (pixels) = ({x_offset}, {y_offset}), move (mm) = ({x_move:.3f}, {y_move:.3f})")
        
        print("Failed to center lower camera in upper camera view after maximum iterations")
        self.send_gcode_command("M106 P3 S0") #Turn off LED Ring for upper camera
        self.send_gcode_command("T-1", check=False)
        return False




    
        

if __name__ == "__main__":
    Printer = CalibrateToolheads()
    try:
        #    Printer.home()
        #    Printer.calibrate_with_camera(0)
        Printer.calibrate_tool_with_camera(0,camera = 0)
        Printer.calibrate_tool_with_camera(1,camera = 0)
        Printer.calibrate_tool_with_camera(2,camera = 0)
        Printer.calibrate_camera_with_camera(3,camera= 2)
    finally:
        # Ensure LEDs are turned off before closing
        Printer.send_gcode_command("M106 P3 S0") #Turn off LED Ring for upper camera
        Printer.send_gcode_command("M106 P4 S0") #Turn off LED Ring for lower camera
        Printer.close()
