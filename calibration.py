from dsf.connections import CommandConnection
from vision_tools import VisionTools
import time


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

    def calibrate_with_camera(self, toolhead_number):
        """
        Prepare a specific toolhead for calibration and center it in the camera view.
        Uses computer vision to locate the tool and iteratively moves it to the center.
        Once centered, sets the tool offset relative to the camera position.
        
        Args:
            toolhead_number (int): The number of the toolhead to calibrate
            
        Returns:
            bool: True if calibration successful, False if tool cannot be centered
        """
        # Select the tool and move to initial position
        self.send_gcode_command("T-1", check=False)
        self.send_gcode_command(f"T{toolhead_number}", check=False)
        
        # Move to safe Z height first
        self.send_gcode_command("G0 Z166.41 F6000", check=False)
        
        # Move to approximate camera XY position
        self.send_gcode_command(f"G0 X{self.camera_location[0]} Y{self.camera_location[1]} Z{self.camera_location[2]} F6000", check=False)
        time.sleep(1.5)
        # Constants for the centering algorithm
        MAX_ITERATIONS = 20  # Maximum number of attempts to center
        TOLERANCE = 2  # Pixels from center considered "centered"
        INITIAL_PIXELS_TO_MM = 0.017  # Initial conversion factor
        
        # Start Camera by instantiating VisionTools class
        camera = VisionTools(0)
        
        # Get image dimensions from vision tools and calculate center
        image_width, image_height = camera.get_image_dimensions()
        IMAGE_CENTER = (image_width // 2, image_height // 2)
        print(f"Image center: {IMAGE_CENTER}")
        iteration = 0
        pixels_to_mm = INITIAL_PIXELS_TO_MM  # Start with initial conversion factor
        previous_pos = None
        previous_pixel_pos = None
        
        while iteration < MAX_ITERATIONS:
            # Get tool position in camera image
            tool_pos = camera.find_tool_position()
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
                    self.send_gcode_command("T-1", check=False)
                    return True
                except Exception as e:
                    print(f"Failed to set tool offset: {str(e)}")
                    self.send_gcode_command("T-1", check=False)
                    return False
            
            # Get current machine position
            self.send_gcode_command("M114", check=False)
            current_pos = self.parse_position(self.response)
            
            # Calculate pixels_to_mm based on previous movement if available
            # if previous_pos is not None and previous_pixel_pos is not None:
                # Calculate actual movement in mm
                # dx_mm = current_pos['X'] - previous_pos['X']
                # dy_mm = current_pos['Y'] - previous_pos['Y']
                
                # Calculate pixel movement
                # dx_pixels = previous_pixel_pos[0] - x_pixel
                # dy_pixels = previous_pixel_pos[1] - y_pixel
                
                # Update conversion factors if movement was significant (avoid division by zero or tiny movements)
                # if abs(dx_mm) > 0.1 and abs(dx_pixels) > 2:
                #     pixels_to_mm_x = abs(dx_pixels / dx_mm)
                
                # if abs(dy_mm) > 0.1 and abs(dy_pixels) > 2:
                #     pixels_to_mm_y = abs(dy_pixels / dy_mm)
                    
                # Use average of X and Y conversion factors if both are valid
                # if 'pixels_to_mm_x' in locals() and 'pixels_to_mm_y' in locals():
                #     pixels_to_mm = (pixels_to_mm_x + pixels_to_mm_y) / 2
                #     print(f"Updated pixels_to_mm: {pixels_to_mm:.4f}")
            
            # Store current positions for next iteration
            previous_pos = current_pos
            previous_pixel_pos = (x_pixel, y_pixel)
            
            # Calculate move distance using current conversion factor
            x_move = -y_offset * pixels_to_mm #May need to be modified based on camera orientation
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
        return False




    
        

if __name__ == "__main__":
    Printer = CalibrateToolheads()
#    Printer.home()
#    Printer.calibrate_with_camera(0)
    Printer.calibrate_with_camera(0)
    Printer.calibrate_with_camera(1)
    Printer.calibrate_with_camera(2)
    Printer.close()




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
    
    # Temperature Control
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
    'T-1', 'T0', 'T1', 'T2',
]
