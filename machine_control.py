"""
Machine Control Module for DuetPnP

This module handles all machine control operations including:
- Printer class: G-code commands, motion, tool control, calibration
- Feeder class: Rotary belt component feeding system
- Generic centering function: Uses vision feedback for precise positioning

Integrates with machine_vision module for camera configuration and coordinate transforms.
All optimizations from calibration.py and feed.py are preserved.
"""

from dsf.connections import CommandConnection
from machine_vision import VisionTools, CameraConfig, show_frame_with_overlay
import time
import json
import cv2
from typing import Dict, List, Tuple, Optional, Callable

# Valid G-code commands whitelist for validation
VALID_GCODES = [
    'G0', 'G1', 'G2', 'G3', 'G4', 'G28',
    'G29', 'G30', 'G31', 'G32',
    'G38.2', 'G38.3', 'G38.4', 'G38.5',
    'G17', 'G18', 'G19', 'G20', 'G21',
    'G53', 'G54', 'G55', 'G56', 'G57', 'G58', 'G59',
    'G59.1', 'G59.2', 'G59.3', 'G60', 'G68', 'G69',
    'G90', 'G91', 'G92', 'G10', 'G11',
    'M0', 'M1', 'M3', 'M4', 'M5',
    'M17', 'M18', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25',
    'M26', 'M27', 'M28', 'M29', 'M30', 'M32', 'M36', 'M37',
    'M38', 'M39', 'M42',
    'M104', 'M105', 'M106', 'M107', 'M108', 'M109',
    'M116', 'M140', 'M141', 'M143', 'M144', 'M190', 'M191',
    'M111', 'M114', 'M115', 'M119', 'M122',
    'M280', 'M290', 'M291', 'M292', 'M300',
    'M500', 'M501', 'M502', 'M503', 'M505',
    'M409', 'M550', 'M551', 'M552', 'M553', 'M554',
    'M555', 'M556', 'M557', 'M558', 'M559', 'M560',
    'M563', 'M567', 'M568', 'M569',
    'M569.1', 'M569.2', 'M569.3', 'M569.4', 'M569.5', 'M569.6', 'M569.7',
    'M201', 'M201.1', 'M203', 'M204', 'M205', 'M206', 'M207', 'M208',
    'M220', 'M221', 'M566', 'M567', 'M568', 'M569', 'M570',
    'M581', 'M582', 'M584', 'M585', 'M586', 'M587', 'M588', 'M589',
    'M591', 'M592', 'M593', 'M594', 'M595',
    'M80', 'M81', 'M112', 'M999',
    'M600', 'M701', 'M702', 'M703',
    'T-1', 'T0', 'T1', 'T2', 'T3',
]



class Printer:
    """
    Printer control class for Duet3D-based pick-and-place system.
    
    Handles:
    - G-code command execution and validation
    - Position querying and parsing
    - Tool offset management
    - Vacuum and solenoid control
    - Motion commands
    - Tool calibration operations
    
    Optimized with vision instance caching and reduced debug verbosity.
    """
    
    def __init__(self, upward_camera_number: int = 0, debug: bool = False):
        """
        Initialize printer connection.

        Args:
            upward_camera_number: Camera number for the upward-facing camera (default: 0)
            debug: Enable verbose debug printing
        """
        self.connection = CommandConnection(debug=False)
        self.connection.connect()
        self.debug = debug
        self._vision_cache = {}  # Cache VisionTools instances
        self.response = ""
        self.upward_camera_number = upward_camera_number  # Store for later use

        # Load camera configurations (location and LED pins)
        self._load_camera_configs(upward_camera_number)

        # Load machine configuration (vacuum, solenoid, etc.)
        self._load_machine_config()

        print("Printer connection established")

    def _load_camera_configs(self, upward_camera_number: int):
        """
        Load camera configurations from JSON files.

        Loads camera location and LED pin mappings for all available cameras.

        Args:
            upward_camera_number: Camera number for the upward-facing camera
        """
        # Dictionary to store LED pin mappings: {camera_number: pin}
        self.camera_led_pins = {}

        # Load all available camera configs (try common camera numbers)
        for cam_num in [0, 1, 2, 3]:
            try:
                config_path = f"config/camera_config_{cam_num}.json"
                with open(config_path, 'r') as f:
                    camera_config = json.load(f)

                # Store LED pin mapping
                led_control = camera_config.get('led_control', {})
                if 'pin' in led_control:
                    self.camera_led_pins[cam_num] = led_control['pin']
                    if self.debug:
                        print(f"Loaded LED pin for camera {cam_num}: {led_control['pin']}")

                # Load camera location only for upward-facing camera
                if cam_num == upward_camera_number:
                    loc = camera_config.get('machine_location', {})
                    if all(loc.get(k) is not None for k in ['x', 'y', 'z']):
                        self.camera_location = [loc['x'], loc['y'], loc['z']]
                        print(f"Loaded camera location from {config_path}: {self.camera_location}")
                    else:
                        raise ValueError(f"Camera {upward_camera_number} does not have a valid machine_location")

            except FileNotFoundError:
                # Expected for cameras that don't exist
                continue
            except Exception as e:
                if cam_num == upward_camera_number:
                    print(f"Warning: Could not load camera {cam_num} config: {e}")

        # Fallback for camera location if not loaded
        if not hasattr(self, 'camera_location'):
            print(f"Warning: Could not load camera location for camera {upward_camera_number}, using default")
            self.camera_location = [21.6, -59.7, 168.00]

        # Fallback LED pins if no configs found
        if not self.camera_led_pins:
            print("Warning: No camera LED pin configs loaded, using defaults")
            self.camera_led_pins = {0: "fan4", 2: "fan3"}

    def _load_machine_config(self):
        """
        Load machine configuration from machine_config.json.

        Loads vacuum pin, solenoid pin, and other machine-specific settings.
        """
        try:
            with open('config/machine_config.json', 'r') as f:
                machine_config = json.load(f)

            # Load vacuum settings
            vacuum_config = machine_config.get('vacuum', {})
            self.vacuum_pin = vacuum_config.get('pin', 'fan1')
            self.vacuum_on_value = vacuum_config.get('on_value', 40)
            self.vacuum_off_value = vacuum_config.get('off_value', 0)

            # Load solenoid settings
            solenoid_config = machine_config.get('solenoid', {})
            self.solenoid_pin = solenoid_config.get('pin', 'fan2')
            self.solenoid_on_value = solenoid_config.get('on_value', 255)
            self.solenoid_off_value = solenoid_config.get('off_value', 0)

            if self.debug:
                print(f"Loaded machine config: vacuum={self.vacuum_pin}, solenoid={self.solenoid_pin}")

        except FileNotFoundError:
            print("Warning: config/machine_config.json not found, using defaults")
            self.vacuum_pin = 'fan1'
            self.vacuum_on_value = 40
            self.vacuum_off_value = 0
            self.solenoid_pin = 'fan2'
            self.solenoid_on_value = 255
            self.solenoid_off_value = 0
        except Exception as e:
            print(f"Warning: Could not load machine config: {e}, using defaults")
            self.vacuum_pin = 'fan1'
            self.vacuum_on_value = 40
            self.vacuum_off_value = 0
            self.solenoid_pin = 'fan2'
            self.solenoid_on_value = 255
            self.solenoid_off_value = 0

    def send_gcode_command(self, gcode_command: str, check: bool = True):
        """Send G-code command to printer with optional validation."""
        if check:
            if not isinstance(gcode_command, str):
                raise ValueError("G-code command must be a string")
            base_command = gcode_command.split()[0]
            if base_command not in VALID_GCODES:
                raise ValueError(f"Invalid G-code: {base_command}")
        self.response = self.connection.perform_simple_code(gcode_command)

    def wait_for_idle(self, timeout: float = 10) -> bool:
        """
        Wait for the printer to become idle (finish current command).

        Uses M409 K'state.status' to query printer status and waits until
        the response contains "idle".

        Args:
            timeout: Maximum time to wait in seconds (default: 10)

        Returns:
            bool: True if printer became idle, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Query printer status
            self.send_gcode_command("M409 K'state.status'", check=False)
            # Check if printer is idle
            if "idle" in self.response.lower():
                if self.debug:
                    print('Printer idle, continuing')
                return True
            time.sleep(0.05)

        print(f"Warning: Printer did not become idle within {timeout}s timeout")
        return False

    def send_gcode_and_wait(self, gcode_command: str, check: bool = True, timeout: float = 10) -> bool:
        """
        Send G-code command and wait for printer to become idle.

        Combines send_gcode_command() with wait_for_idle() for reliable
        execution of commands that need completion before next operation.

        Args:
            gcode_command: G-code command to send
            check: Validate G-code command (default: True)
            timeout: Maximum time to wait for idle in seconds (default: 10)

        Returns:
            bool: True if command completed and printer became idle, False if timeout
        """
        self.send_gcode_command(gcode_command, check=check)
        return self.wait_for_idle(timeout)

    def close(self):
        """Close printer connection."""
        self.connection.close()
        print("Printer connection closed")
    
    def home(self):
        """Home all axes."""
        self.send_gcode_command("T-1", check=False)
        self.send_gcode_command("G28", check=False)

    def parse_position(self, response: str) -> Dict[str, float]:
        """Parse M114 response to get current position."""
        if not response:
            raise ValueError("Empty response from M114")
        position = {}
        valid_axes = ['X', 'Y', 'Z', 'U', 'V', 'W', 'A']
        required_axes = ['X', 'Y', 'Z']
        for part in response.split():
            if ':' in part:
                axis, value = part.split(':')
                if axis in valid_axes:
                    position[axis] = float(value)
        missing = set(required_axes) - set(position.keys())
        if missing:
            raise ValueError(f"Missing axes: {missing}")
        return position

    def parse_tool_offsets(self, response: str, toolhead_number: int) -> Dict[str, float]:
        """
        Parse G10 response to get tool offsets.

        Example response: "Tool 0: offsets X1.300 Y-102.200 Z-18.250 U0.000 V0.000, active/standby temperature(s) 0.0/0.0"

        Args:
            response: Response string from G10 command
            toolhead_number: Tool number to verify response matches

        Returns:
            dict: Dictionary containing axis offsets (X, Y, Z, and optionally U, V, W, A)

        Raises:
            ValueError: If response is empty, tool number doesn't match, or parsing fails
        """
        if not response:
            raise ValueError("Empty response from G10 command")

        offsets = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}  # Default to zero offsets
        valid_axes = ['X', 'Y', 'Z', 'U', 'V', 'W', 'A']

        try:
            # Check if this is the correct tool response
            if not response.startswith(f"Tool {toolhead_number}:"):
                raise ValueError(f"Response does not match Tool {toolhead_number}")

            # Extract the offsets section
            if "offsets " not in response:
                raise ValueError("No 'offsets' section found in response")

            offset_section = response.split("offsets ")[1].split(",")[0]

            # Parse each axis offset
            for axis in valid_axes:
                # Look for axis in the response
                axis_start = offset_section.find(axis)
                if axis_start != -1:
                    # Extract value until next space or end
                    value_str = ""
                    i = axis_start + 1
                    while i < len(offset_section) and (offset_section[i].isdigit() or offset_section[i] in '.-'):
                        value_str += offset_section[i]
                        i += 1
                    try:
                        offsets[axis] = float(value_str)
                    except ValueError:
                        raise ValueError(f"Invalid offset value for {axis}: {value_str}")

        except Exception as e:
            raise ValueError(f"Error parsing G10 response: {str(e)}")

        if self.debug:
            print(f"Parsed Tool {toolhead_number} offsets: {offsets}")

        return offsets

    def get_tool_offsets(self, toolhead_number: int) -> Dict[str, float]:
        """
        Get current tool offsets from firmware.

        Args:
            toolhead_number: Tool number (0, 1, 2, 3, etc.)

        Returns:
            dict: Dictionary containing axis offsets
        """
        self.send_gcode_command(f"G10 P{toolhead_number}", check=False)
        return self.parse_tool_offsets(self.response, toolhead_number)

    def parse_axis_mapping(self, response: str) -> Dict[str, str]:
        """
        Parse M563 response to get tool axis mappings.

        Example response: "Tool 0 - drives: 0; heaters (active/standby temps): 2 (0.0/0.0); xmap: X; ymap: Y; zmap: Z; fans: 0; no spindle; status: standby"

        Args:
            response: Response string from M563 command

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
        if not all(axis in mappings for axis in ['X', 'Y', 'Z']):
            raise ValueError(f"Missing required axis mappings in M563 response. Got: {mappings}")

        return mappings

    def get_current_position(self) -> Dict[str, float]:
        """Get current machine position."""
        self.wait_for_idle()
        self.send_gcode_command("M114", check=False)
        return self.parse_position(self.response)
    
    def linear_move(self, x: Optional[float] = None, y: Optional[float] = None, 
                   z: Optional[float] = None, feed_rate: int = 6000, f: int = 6000):
        """Move to specified position."""
        self.send_gcode_command("G90", check=False)
        if f !=6000:
            feed_rate = f
        move_cmd = "G0"
        if x is not None:
            move_cmd += f" X{x}"
        if y is not None:
            move_cmd += f" Y{y}"
        if z is not None:
            move_cmd += f" Z{z}"
        move_cmd += f" F{feed_rate}"
        self.send_gcode_command(move_cmd, check=False)
        self.wait_for_idle()
    def control_led(self, camera_number: int, state: bool):
        """
        Control camera LED ring light.

        Loads LED pin from camera configuration. If camera not found,
        silently returns without error.

        Args:
            camera_number: Camera number (0, 2, etc.)
            state: True to turn on LED, False to turn off
        """
        # Get LED pin from loaded configs
        pin = self.camera_led_pins.get(camera_number)
        if not pin:
            if self.debug:
                print(f"Warning: No LED pin configured for camera {camera_number}")
            return

        value = 255 if state else 0
        # Extract pin number from format like "fan4" -> "4"
        pin_num = pin.replace('fan', '')
        self.send_gcode_command(f"M106 P{pin_num} S{value}", check=False)
        time.sleep(0.25)
    def control_vacuum(self, state: bool):
        """
        Control vacuum pump.

        Uses vacuum pin and values from machine configuration.

        Args:
            state: True to turn on vacuum, False to turn off
        """
        value = self.vacuum_on_value if state else self.vacuum_off_value
        pin_num = self.vacuum_pin.replace('fan', '')
        self.send_gcode_command(f"M106 P{pin_num} S{value}", check=False)
        # Add delay for vacuum pump to reach full pressure
        time.sleep(1.0)

    def control_solenoid(self, state: bool):
        """
        Control solenoid valve.

        Uses solenoid pin and values from machine configuration.

        Args:
            state: True to open valve, False to close
        """
        value = self.solenoid_on_value if state else self.solenoid_off_value
        pin_num = self.solenoid_pin.replace('fan', '')
        self.send_gcode_command(f"M106 P{pin_num} S{value}", check=False)
        time.sleep(0.5)

    def rotate_c_axis(self, angle_delta: float, feed_rate: int = 6000):
        """
        Rotate C-axis by a relative angle.

        The C-axis rotates the component held by the PnP tool for proper
        orientation before placement. Uses relative positioning.

        Args:
            angle_delta: Angle to rotate in degrees (relative, can be positive or negative)
            feed_rate: Rotation speed in degrees/min (default: 6000)
        """
        self.send_gcode_command("G91", check=False)  # Relative positioning
        self.send_gcode_command(f"G0 C{angle_delta} F{feed_rate}", check=False)
        self.send_gcode_command("G90", check=False)  # Back to absolute positioning

    def select_tool(self, tool_number: int):
        """Select tool by number."""
        self.send_gcode_command(f"T{tool_number}", check=False)
        # Add delay for tool changes (especially camera tool T3)
        self.wait_for_idle()


class Feeder:
    """
    Component feeder control for rotary belt system.
    
    Manages 3-position rotary feeder with B-axis control.
    Uses rock-back-then-forward feed mechanism.
    """
    
    def __init__(self, num_belts: int = 3, radius: float = 11.45):
        """Initialize feeder parameters."""
        if not isinstance(num_belts, int) or num_belts < 1:
            raise ValueError("num_belts must be positive integer")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be positive number")
        
        self.radius = radius
        self.num_belts = num_belts
        self.belt = None  # Current belt position (set after homing)
    
    def home(self, printer: 'Printer'):
        """
        Home feeder by rotating until stall.
        User tests feed positions to identify which belt feeds successfully.
        """
        print("Rotating feeder until stall...")
        printer.send_gcode_command("G91", check=False)  # Relative mode
        printer.send_gcode_command("G1 B120 F900", check=False)  # Rotate until stall
        printer.send_gcode_command("G0 B-5 F900", check=False)  # Back up slightly
        printer.send_gcode_command("G90", check=False)  # Absolute mode
        
        print(f"\nPress 'c' to test feed, then enter belt number (0-{self.num_belts-1})")
        print("Counting from left to right")
        
        while True:
            key = input().lower()
            if key == 'c':
                # Test feed
                theta = (6/self.radius) * (180/3.14159)
                printer.send_gcode_command("G91", check=False)
                printer.send_gcode_command(f"G1 B-{theta} F60000", check=False)
                printer.send_gcode_command(f"G1 B{theta} F400", check=False)
                printer.send_gcode_command("G90", check=False)
                print(f"Press 'c' again or enter belt number (0-{self.num_belts-1})")
            else:
                try:
                    belt_num = int(key)
                    if 0 <= belt_num < self.num_belts:
                        self.belt = belt_num
                        print(f"Feeder homed to belt {belt_num}")
                        return belt_num
                    else:
                        print(f"Invalid. Must be 0-{self.num_belts-1}")
                except ValueError:
                    print("Invalid input")
    
    def feed(self, belt: int, printer: 'Printer'):
        """
        Feed component from specified belt position.
        
        Args:
            belt: Belt number to feed from (0 to num_belts-1)
            printer: Printer instance for G-code commands
        """
        if not isinstance(belt, int):
            raise TypeError("belt must be an integer")
        
        # Calculate rotation if homed
        if self.belt is not None and 0 <= belt < self.num_belts:
            belt_diff = abs(belt - self.belt)
            if belt_diff == 1:
                rotation = -120
            elif belt_diff == 2:
                rotation = -240
            else:
                rotation = 0
            
            if rotation != 0:
                printer.send_gcode_command("G91", check=False)
                printer.send_gcode_command(f"G1 B{rotation} F6000", check=False)
                printer.send_gcode_command("G90", check=False)
                time.sleep(4)
                self.belt = belt
        
        # Execute feed (rock back then push forward)
        theta = (6/self.radius) * (180/3.14159)
        printer.send_gcode_command("G91", check=False)
        printer.send_gcode_command(f"G1 B-{theta} F60000", check=False)  # Back up
        time.sleep(1)
        printer.send_gcode_command(f"G1 B{theta} F400", check=False)  # Feed forward
        printer.send_gcode_command("G90", check=False)


def center_target_in_camera(
    printer: Printer,
    vision: VisionTools,
    camera_config: CameraConfig,
    detection_method: Callable,
    tolerance: int = 2,
    max_iterations: int = 20,
    feed_rate: int = 300,
    debug: bool = False,
    show_display: bool = False
) -> Tuple[bool, Dict[str, float]]:
    """
    Generic centering algorithm using vision feedback.

    Works for any vision-detected target:
    - Tool calibration (detection_method = vision.find_tool_position)
    - Component detection (detection_method = lambda: vision.find_component(template))
    - Any other centering task

    Uses camera_config for coordinate transforms, eliminating hardcoded transforms.

    Args:
        printer: Printer instance for motion control
        vision: VisionTools instance for detection
        camera_config: CameraConfig for coordinate transforms
        detection_method: Callable that returns (pos, angle, frame) or (pos, angle)
        tolerance: Pixels from center considered "centered" (default 2)
        max_iterations: Maximum centering attempts (default 20)
        feed_rate: Movement feed rate in mm/min (default 1200)
        debug: Enable verbose debug output (default False)
        show_display: Show visual feedback window (default False)

    Returns:
        (success: bool, final_position: Dict[str, float])
    """
    image_center = vision.get_image_center()
    iteration = 0
    window_name = "Centering Progress"

    if debug:
        print(f"Starting centering: tolerance={tolerance}px, max_iter={max_iterations}")
        print(f"Image center: {image_center}")

    try:
        while iteration < max_iterations:
            # Detect target
            time.sleep(0.33)
            result = detection_method()

            # Handle different return formats: (pos, angle, frame) or (pos, angle) or (pos,)
            frame = None
            detected_pos = None

            if result is None:
                if debug:
                    print(f"Iteration {iteration}: Target not detected")
                iteration += 1
                continue

            if isinstance(result, tuple):
                if len(result) == 3:
                    # Format: (pos, angle, frame)
                    detected_pos, _, frame = result
                elif len(result) == 2:
                    # Could be (pos, angle) or (x, y)
                    if isinstance(result[0], (dict, tuple)):
                        # Format: (pos, angle)
                        detected_pos = result[0]
                    else:
                        # Format: (x, y)
                        detected_pos = result

            if detected_pos is None:
                if debug:
                    print(f"Iteration {iteration}: Could not parse detection result")
                iteration += 1
                continue

            # Extract x, y from detected position (handle dict or tuple)
            if isinstance(detected_pos, dict):
                x_pixel = detected_pos.get('X')
                y_pixel = detected_pos.get('Y')
                # Validate that dict values are not None
                if x_pixel is None or y_pixel is None:
                    if debug:
                        print(f"Iteration {iteration}: Missing X or Y in dict: {detected_pos}")
                    iteration += 1
                    continue
            elif isinstance(detected_pos, tuple) and len(detected_pos) == 2:
                x_pixel, y_pixel = detected_pos
            else:
                if debug:
                    print(f"Iteration {iteration}: Unexpected position format: {detected_pos}")
                iteration += 1
                continue

            # Calculate pixel offsets from center
            x_pixel_offset = image_center[0] - x_pixel
            y_pixel_offset = image_center[1] - y_pixel
            printer.wait_for_idle()
            
            # Show visual feedback if enabled
            if show_display and frame is not None:
                status_text = f"Iter {iteration+1}/{max_iterations} | Offset: ({x_pixel_offset:.1f}, {y_pixel_offset:.1f})px"
                show_frame_with_overlay(
                    frame=frame,
                    detected_pos=(int(x_pixel), int(y_pixel)),
                    center_pos=(int(image_center[0]), int(image_center[1])),
                    window_name=window_name,
                    text=status_text
                )
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nDisplay closed by user")
                    cv2.destroyWindow(window_name)
                    final_pos = printer.get_current_position()
                    return (False, final_pos)

            # Check if centered
            if abs(x_pixel_offset) <= tolerance and abs(y_pixel_offset) <= tolerance:
                if debug:
                    print(f"Target centered! Final offset: ({x_pixel_offset}, {y_pixel_offset}) pixels")
                if show_display:
                    # Show final centered frame for 1 second
                    if frame is not None:
                        show_frame_with_overlay(
                            frame=frame,
                            detected_pos=(int(x_pixel), int(y_pixel)),
                            center_pos=(int(image_center[0]), int(image_center[1])),
                            window_name=window_name,
                            text="CENTERED!"
                        )
                        cv2.waitKey(1000)
                    cv2.destroyWindow(window_name)
                final_pos = printer.get_current_position()
                return (True, final_pos)

            # Transform pixel offsets to machine movements using camera config
            x_move, y_move = camera_config.pixel_to_machine_movement(
                x_pixel_offset, y_pixel_offset
            )

            # Get current position and calculate new position
            current_pos = printer.get_current_position()
            new_x = current_pos['X'] + x_move
            new_y = current_pos['Y'] + y_move

            # Move to new position
            printer.linear_move(x=new_x, y=new_y, feed_rate=feed_rate)
            printer.wait_for_idle()

            if debug:
                print(f"Iteration {iteration}: pixel_offset=({x_pixel_offset:.1f}, {y_pixel_offset:.1f}), "
                      f"move=({x_move:.3f}, {y_move:.3f})mm")

            # CRITICAL: Clear camera buffer after movement
            # The buffer contains frames from during/right after movement
            time.sleep(0.3)  # Allow mechanical settling
            for _ in range(5):
                vision.capture_frame()
            time.sleep(0.1)  # One more delay for fresh frame

            iteration += 1

        print(f"Failed to center after {max_iterations} iterations")
        if show_display:
            cv2.destroyWindow(window_name)
        final_pos = printer.get_current_position()
        return (False, final_pos)

    except Exception as e:
        # Clean up display on error
        if show_display:
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
        raise e
