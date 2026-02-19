"""
PnP Operations Module for DuetPnP

This module handles PnP-specific operations including:
- ConfigManager: Loads placement configs and component templates
- PnPWorkflow: Orchestrates pick-and-place operations

Uses machine_control and machine_vision modules for hardware interaction.
"""

import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from machine_control import Printer, Feeder, center_target_in_camera
from machine_vision import VisionTools, CameraConfig, load_camera_config, create_display_window, display_image
import time
import os
from datetime import datetime


class ConfigManager:
    """
    Manages all configuration files for PnP operations.

    Handles:
    - Placement configuration (config/placement_config.json)
    - Component templates (upper and lower)
    """

    def __init__(self, config_file: str = "config/placement_config.json"):
        """
        Load configuration from file.

        Args:
            config_file: Path to placement configuration JSON
        """
        self.config_file = config_file
        self.config = None
        self.upper_templates = {}
        self.lower_templates = {}

        self.load_config()
    
    def load_config(self):
        """
        Load placement configuration from JSON file.

        Expected format:
        {
            "components": [
                {
                    "type": "component_name",
                    "upper_template": "templates/component_above.png",
                    "lower_template": "templates/component_below.png",
                    "feed_number": 0,
                    "reel_location": {"x": 0, "y": 0, "z": 0},
                    "reel_focus": 101.25,
                    "feeder_button_location": {"x": 0, "y": 0, "z": 0},
                    "placements": [
                        {"x": 0, "y": 0, "z": 0, "rotation": 0}
                    ]
                }
            ]
        }

        Note: Vacuum and solenoid pins are configured in config/machine_config.json
        Note: feeder_button_location is where the tool presses to advance the feeder
        """
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            
            # Load templates for each component
            for component in self.config['components']:
                comp_type = component['type']
                
                # Load upper template (feeder view)
                upper_path = component['upper_template'] #"From Above, with upper camera (which is downward facing)
                upper_template = cv2.imread(upper_path, cv2.IMREAD_GRAYSCALE)
                if upper_template is None:
                    raise ValueError(f"Could not load upper template: {upper_path}")
                self.upper_templates[comp_type] = upper_template
                
                # Load lower template (tool view)
                lower_path = component['lower_template']
                lower_template = cv2.imread(lower_path, cv2.IMREAD_GRAYSCALE)
                if lower_template is None:
                    raise ValueError(f"Could not load lower template: {lower_path}")
                self.lower_templates[comp_type] = lower_template
            
            print(f"Loaded config: {len(self.config['components'])} component types")
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            raise ValueError(f"Error loading configuration: {str(e)}")

    def get_component(self, component_type: str) -> Optional[Dict]:
        """Get component configuration by type."""
        for component in self.config['components']:
            if component['type'] == component_type:
                return component
        return None
    
    def get_all_components(self) -> List[Dict]:
        """Get all component configurations."""
        return self.config['components']


class PnPWorkflow:
    """
    Orchestrates pick-and-place workflow operations.
    
    Handles:
    - Component pickup from feeder
    - Component orientation detection and correction
    - Component placement
    - Placement verification
    """
    
    def __init__(self, printer: Printer, feeder: Feeder, config_manager: ConfigManager,
                 camera_configs: Dict[int, CameraConfig], placement_offset: Optional[Dict[str, float]] = None,
                 yolo_model_path: Optional[str] = None, use_yolo: bool = True):
        """
        Initialize PnP workflow.

        Args:
            printer: Printer instance
            feeder: Feeder instance
            config_manager: ConfigManager instance
            camera_configs: Dictionary mapping camera numbers to CameraConfig instances
            placement_offset: Optional offset dict with 'x' and 'y' keys (in mm) to apply to all placements
            yolo_model_path: Optional path to YOLO-OBB model for faster detection
            use_yolo: If True and model available, use YOLO instead of template matching
        """
        self.printer = printer
        self.feeder = feeder
        self.config = config_manager
        self.camera_configs = camera_configs
        self.placement_offset = placement_offset if placement_offset else {'x': 0.0, 'y': 0.0}

        # Initialize vision tools for each camera
        self.vision_upper = VisionTools(2, target='tool',
                                       camera_config=camera_configs[2])
        self.vision_lower = VisionTools(0, target='tool',
                                       camera_config=camera_configs[0])

        # YOLO detection settings
        self.use_yolo = use_yolo and yolo_model_path is not None
        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                self.vision_upper.load_yolo_model(yolo_model_path, imgsz=320, conf=0.5)
                self.vision_lower.load_yolo_model(yolo_model_path, imgsz=320, conf=0.5)
                print(f"✓ YOLO detection enabled (model: {yolo_model_path})")
            except Exception as e:
                print(f"⚠️ Failed to load YOLO model: {e}")
                print("  Falling back to template matching")
                self.use_yolo = False
        elif yolo_model_path:
            print(f"⚠️ YOLO model not found: {yolo_model_path}")
            print("  Using template matching instead")
            self.use_yolo = False

        # Centering parameters
        self.TOLERANCE = 1
        self.MAX_ITERATIONS = 10

        # Checkpoint state
        self.checkpoint_file = "config/pnp_checkpoint.json"
        self.current_component_index = 0
        self.current_placement_index = 0
        self.completed_placements = []
        self.failed_placements = []

        # Session management for photos
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join("sessions", self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"PnP Workflow initialized - Session: {self.session_id}")
        print(f"Photos will be saved to: {self.session_dir}")

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        from datetime import datetime
        checkpoint_data = {
            "description": "PnP workflow checkpoint for resume capability",
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "status": "paused",
            "config_file": self.config.config_file,
            "session_id": self.session_id,
            "current_component_index": self.current_component_index,
            "current_placement_index": self.current_placement_index,
            "completed_placements": self.completed_placements,
            "failed_placements": self.failed_placements
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"✓ Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            print(f"Warning: Could not save checkpoint: {e}")

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint from file and restore state.

        Returns:
            bool: True if checkpoint loaded successfully, False otherwise
        """
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            if checkpoint_data.get('status') == 'idle':
                return False  # No checkpoint to resume

            # Verify config file matches
            checkpoint_config = checkpoint_data.get('config_file')
            if checkpoint_config and checkpoint_config != self.config.config_file:
                print(f"Warning: Checkpoint was for different config file:")
                print(f"  Checkpoint: {checkpoint_config}")
                print(f"  Current:    {self.config.config_file}")
                response = input("Continue anyway? [y/N]: ").strip().lower()
                if response != 'y':
                    print("Checkpoint load cancelled")
                    return False

            self.current_component_index = checkpoint_data.get('current_component_index', 0)
            self.current_placement_index = checkpoint_data.get('current_placement_index', 0)
            self.completed_placements = checkpoint_data.get('completed_placements', [])
            self.failed_placements = checkpoint_data.get('failed_placements', [])

            print(f"✓ Checkpoint loaded from {self.checkpoint_file}")
            print(f"  Config file: {checkpoint_config or 'unknown'}")
            if checkpoint_data.get('session_id'):
                print(f"  Session ID: {checkpoint_data['session_id']}")
            print(f"  Resuming from component {self.current_component_index}, placement {self.current_placement_index}")
            print(f"  Completed: {len(self.completed_placements)} placements")
            if self.failed_placements:
                print(f"  Failed: {len(self.failed_placements)} placements")
            return True

        except FileNotFoundError:
            print(f"No checkpoint file found, starting from beginning")
            return False
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return False

    def clear_checkpoint(self):
        """Clear checkpoint file after successful completion."""
        checkpoint_data = {
            "description": "PnP workflow checkpoint for resume capability",
            "version": "1.0",
            "last_updated": None,
            "status": "idle",
            "current_component_index": 0,
            "current_placement_index": 0,
            "completed_placements": [],
            "failed_placements": []
        }
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not clear checkpoint: {e}")

    def run(self, resume: bool = False):
        """
        Execute full PnP workflow for all components in configuration.

        Args:
            resume: If True, attempt to resume from checkpoint
        """
        print("\nStarting PnP workflow...")
        create_display_window()

        # Load checkpoint if resuming
        if resume:
            self.load_checkpoint()

        components = self.config.get_all_components()

        # Iterate through components starting from checkpoint
        for comp_idx in range(self.current_component_index, len(components)):
            component = components[comp_idx]
            self.current_component_index = comp_idx
            comp_type = component['type']
            print(f"\n{'='*60}")
            print(f"Component {comp_idx + 1}/{len(components)}: {comp_type}")
            print(f"{'='*60}")

            # Determine starting placement index
            start_placement = self.current_placement_index if comp_idx == self.current_component_index else 0

            for place_idx in range(start_placement, len(component['placements'])):
                placement = component['placements'][place_idx]
                self.current_placement_index = place_idx

                placement_id = f"{comp_type}_{comp_idx}_{place_idx}"
                print(f"\n  Placement {place_idx + 1}/{len(component['placements'])}: ({placement['x']}, {placement['y']}, {placement['z']})")

                # Try to place component with error handling
                try:
                    success = self.place_component(component, placement, place_idx + 1)

                    if success:
                        self.completed_placements.append(placement_id)
                        self.save_checkpoint()
                    else:
                        # Placement failed - ask user what to do
                        action = self._handle_placement_error(placement_id, "Placement failed")
                        if action == 'abort':
                            print("\n✗ Workflow aborted by user")
                            self.save_checkpoint()
                            return False
                        elif action == 'retry':
                            # Retry same placement
                            place_idx -= 1
                            self.current_placement_index -= 1
                            continue
                        elif action == 'skip':
                            self.failed_placements.append(placement_id)
                            self.save_checkpoint()
                            continue

                except Exception as e:
                    # Unexpected error occurred
                    print(f"\n✗ Error during placement: {e}")
                    import traceback
                    traceback.print_exc()

                    action = self._handle_placement_error(placement_id, str(e))
                    if action == 'abort':
                        print("\n✗ Workflow aborted by user")
                        self.save_checkpoint()
                        return False
                    elif action == 'retry':
                        # Retry same placement (don't increment index)
                        continue
                    elif action == 'skip':
                        self.failed_placements.append(placement_id)
                        self.save_checkpoint()
                        continue

            # Reset placement index for next component
            self.current_placement_index = 0

        print("\n" + "="*60)
        print("✓ PnP workflow complete!")
        print(f"  Completed: {len(self.completed_placements)} placements")
        if self.failed_placements:
            print(f"  Failed: {len(self.failed_placements)} placements")
            print(f"  Failed IDs: {', '.join(self.failed_placements)}")
        print("="*60)

        # Clear checkpoint on successful completion
        self.clear_checkpoint()
        return True

    def capture_verification_photo(self, component_type: str, placement_number: int, placement_pos: Dict) -> bool:
        """
        Capture verification photo of placed component using upper camera (Tool 3).

        Args:
            component_type: Type of component (e.g., "resistor_0201")
            placement_number: Sequential placement number
            placement_pos: Placement position dict with x, y, z

        Returns:
            bool: True if photo captured successfully
        """
        print("  5. Capturing verification photo...")

        # Get camera 2 (upper camera) focus distance from config
        camera2_config = self.camera_configs[2]
        focus_dist_config = camera2_config.config.get('focus_distance', {})
        focus_distance = focus_dist_config.get('value', 101.25) if isinstance(focus_dist_config, dict) else focus_dist_config

        # Photo focus height = focus distance from bed + placement Z value
        photo_focus_height = focus_distance + placement_pos['z']

        # Switch to camera tool
        # Use fast tool change if already on Tool 2 or 3 (no physical tool change needed)
        current_tool = self.printer.get_current_tool()
        use_fast = current_tool in [2, 3]
        self.printer.select_tool(3, fast=use_fast)
        if not use_fast:
            self.printer.select_tool(3)  # Camera tool changetime (only needed for physical change)

        # Move to placement location at focus height
        self.printer.linear_move(z=150)  # Safe height first
        self.printer.control_led(2, True)
        self.printer.linear_move(
            x=placement_pos['x'],
            y=placement_pos['y'],
            z=photo_focus_height
        )
        self.printer.wait_for_idle()

        # Turn on upper camera LED
        
        #time.sleep(0.3)  # Let camera adjust

        # Clear camera buffer after movement and LED turn-on
        for _ in range(5):
            self.vision_upper.capture_frame()
        #time.sleep(0.1)

        # Capture frame
        frame = self.vision_upper.capture_frame()

        # Turn off LED
        self.printer.control_led(2, False)

        if frame is None:
            print("  Warning: Failed to capture verification photo")
            return False

        # Save photo with naming: component_type_#.png
        photo_filename = f"{component_type}_{placement_number}.png"
        photo_path = os.path.join(self.session_dir, photo_filename)

        success = cv2.imwrite(photo_path, frame)
        if success:
            print(f"  ✓ Photo saved: {photo_filename}")
            return True
        else:
            print(f"  Warning: Failed to save photo to {photo_path}")
            return False

    def _handle_placement_error(self, placement_id: str, error_msg: str) -> str:
        """
        Handle placement error with user prompt.

        Args:
            placement_id: Identifier for failed placement
            error_msg: Error message to display

        Returns:
            str: User action ('retry', 'skip', 'abort')
        """
        print(f"\n{'!'*60}")
        print(f"ERROR: {error_msg}")
        print(f"Placement ID: {placement_id}")
        print(f"{'!'*60}")

        while True:
            response = input("\nChoose action:\n  [R]etry this placement\n  [S]kip to next placement\n  [A]bort workflow\n> ").strip().upper()

            if response in ['R', 'RETRY']:
                return 'retry'
            elif response in ['S', 'SKIP']:
                return 'skip'
            elif response in ['A', 'ABORT']:
                return 'abort'
            else:
                print("Invalid choice. Please enter R, S, or A.")

    def press_feeder_button(self, component: Dict,feed_rate: int = 2) -> bool:
        """
        Press the feeder button to advance a new component.

        Args:
            component: Component configuration dictionary
            feed_rate: Number of components fed per button press
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if feeder_button_location is configured

        if not hasattr(self, 'pressed'):
            self.pressed = 1
        elif self.pressed == feed_rate:           
            self.pressed = 1
        else:
            self.pressed += 1
            return True

        if 'feeder_button_location' not in component:
            print("  Warning: No feeder_button_location configured for this component")
            return False

        button_loc = component['feeder_button_location']
        print(f"  Pressing feeder button at X{button_loc['x']:.1f}, Y{button_loc['y']:.1f}")

        # Switch to PnP tool (T2) for pressing button
        # Use fast tool change if already on Tool 2 or 3 (no physical tool change needed)
        current_tool = self.printer.get_current_tool()
        use_fast = current_tool in [2, 3]
        self.printer.select_tool(2, fast=use_fast)
        #self.printer.wait_for_idle()

        # Move to safe height above button
        self.printer.linear_move(z=150)

        # Move to button XY location
        self.printer.linear_move(x=button_loc['x'], y=button_loc['y'])
        self.printer.wait_for_idle()

        # Press down on button
        press_height = button_loc['z']
        self.printer.linear_move(z=press_height)
        self.printer.linear_move(z=press_height-20,f=300)
        self.printer.wait_for_idle()
        self.printer.linear_move(z=press_height)

        #Press again (2x to feed part):
        #self.printer.linear_move(z=press_height,f=600)
        #self.printer.wait_for_idle()
        
        # Lift back up
        self.printer.linear_move(z=150)
        self.printer.wait_for_idle()

        print("  ✓ Feeder button pressed")

        # Wait for feeder to advance (adjust timing as needed)
        

        return True

    def place_component(self, component: Dict, placement: Dict, placement_number: int) -> bool:
        """
        Pick component from feeder and place at target location.

        Args:
            component: Component configuration dictionary
            placement: Placement location dictionary
            placement_number: Sequential placement number for this component type

        Returns:
            True if successful, False otherwise
        """
        # 0. Press feeder button to advance component (if configured)
        if 'feeder_button_location' in component:
            print("  0. Advancing feeder...")
            #self.press_feeder_button(component)

        # 1. Locate component in feeder
        print("  1. Locating component...")
        reel_loc = component['reel_location']
        reel_focus = component.get('reel_focus', 101.25)
        
        # Move upper camera to feeder location
        # Use fast tool change if already on Tool 2 or 3 (no physical tool change needed)
        current_tool = self.printer.get_current_tool()
        use_fast = current_tool in [2, 3]
        if use_fast:
            print(f"    (Fast tool change: current tool is T{current_tool})")
        self.printer.select_tool(3, fast=use_fast)  # Camera tool
        self.printer.control_led(2, True)  # Upper camera LED
        self.printer.linear_move(z=150,f = 6000)  # Safe height
        self.printer.linear_move(x=reel_loc['x']+2, y=reel_loc['y']-2, z=reel_focus, f=12000)
        self.printer.linear_move(x=reel_loc['x'], y=reel_loc['y'], z=reel_focus, f=6000)
        self.printer.wait_for_idle()

        # Clear camera buffer after movement and LED turn-on
        #time.sleep(0.3)  # Allow LED to stabilize and camera to adjust
        for _ in range(5):
            self.vision_upper.capture_frame()
        #time.sleep(0.1)

        # Center component in view
        template_path = component['upper_template']

        def detect_component_upper():
            """Detection method for upper camera that returns frame for display."""
            # Capture fresh frame
            frame = self.vision_upper.capture_frame()
            if frame is None:
                return None, None, None

            # Find component using YOLO or template matching
            if self.use_yolo:
                pos, angle = self.vision_upper.find_component_yolo(expected_angle=0)
            else:
                pos, angle = self.vision_upper.find_component(template_path)
            return pos, angle, frame

        success, final_pos = center_target_in_camera(
            self.printer,
            self.vision_upper,
            self.camera_configs[2],
            detect_component_upper,
            tolerance=2,
            max_iterations=self.MAX_ITERATIONS,
            debug=False,
            show_display=True
        )
        
        if not success:
            print("  Failed to locate component!")
            self.printer.control_led(2, False)
            return False
        
        pickup_pos = final_pos.copy()
        print(f"  Component located at: X{pickup_pos['X']:.2f}, Y{pickup_pos['Y']:.2f}")
        
        # 2. Pick up component
        print("  2. Picking up component...")
        self.printer.control_led(2, False)
        self.printer.select_tool(2,fast=True)  # PnP tool


        #self.printer.linear_move(z=150,f=6000) #Lift to safe height
        self.printer.linear_move(x=pickup_pos['X'], y=pickup_pos['Y'],f=6000)
        #self.printer.wait_for_idle()
        self.printer.control_solenoid(True)
        self.printer.control_vacuum(True)
        #time.sleep(0.25) no longer necessary
        z_pickup_height = component.get('reel_location').get('z', 0)
        self.printer.linear_move(z=z_pickup_height,f=6000) #Move to pickup height
        self.printer.wait_for_idle()
        #time.sleep(0.1) #Ensure Part is picked up.
        self.printer.linear_move(z=50, f=12000) #Lift to safe height
        #self.printer.wait_for_idle()
        
        # 3. Determine orientation with lower camera
        print("  3. Checking component orientation...")
        self.printer.control_led(0, True)  # Lower camera LED
        camera_loc = self.printer.camera_location
        self.printer.linear_move(x=camera_loc[0]+2,y=camera_loc[1]+2,z=camera_loc[2], f=6000) #Move to safe location
        self.printer.linear_move(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2])
        self.printer.wait_for_idle()

        # Clear camera buffer after movement and LED turn-on
        #time.sleep(0.3)  # Allow LED to stabilize
        for _ in range(5):
            self.vision_lower.capture_frame()

        # Detect component and rotation with retry logic and visual feedback
        lower_template_path = component['lower_template']
        desired_angle = placement.get('rotation', 0)

        # Keep trying until component is detected
        max_detection_attempts = 20
        detection_attempt = 0
        detected_angle = None
        center_pos = None

        print("  Waiting for component detection...")
        window_name = "Component Orientation Detection"

        while detection_attempt < max_detection_attempts:
            # Capture frame
            frame = self.vision_lower.capture_frame()

            if frame is not None:
                # Try to find component using YOLO or template matching
                if self.use_yolo:
                    detection_result = self.vision_lower.find_component_yolo(expected_angle=0)
                else:
                    detection_result = self.vision_lower.find_component(lower_template_path)
                pos, angle = detection_result

                # Create display with detection info
                display = frame.copy()
                image_h, image_w = display.shape[:2]
                img_center = (image_w // 2, image_h // 2)

                if pos is not None:
                    # Component detected!
                    center_pos = pos
                    detected_angle = angle

                    # Draw detection visualization
                    cv2.circle(display, pos, 20, (0, 255, 0), 3)
                    cv2.circle(display, pos, 2, (0, 0, 255), -1)

                    # Draw center crosshair
                    cv2.line(display, (img_center[0]-20, img_center[1]),
                            (img_center[0]+20, img_center[1]), (255, 0, 0), 2)
                    cv2.line(display, (img_center[0], img_center[1]-20),
                            (img_center[0], img_center[1]+20), (255, 0, 0), 2)

                    # Add text
                    text = f"DETECTED! Angle: {angle}°, Target: {desired_angle}°"
                    cv2.putText(display, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Press 'c' to continue or 'q' to abort", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.imshow(window_name, display)

                    # Wait for user confirmation
                    key = cv2.waitKey(1000) & 0xFF  # Show for 1 second or until key press
                    if key == ord('c') or key == 13:  # 'c' or Enter
                        print(f"  Component detected at {detected_angle}°, target is {desired_angle}°")
                        cv2.destroyWindow(window_name)
                        break
                    elif key == ord('q') or key == 27:  # 'q' or ESC
                        print("  Detection aborted by user")
                        cv2.destroyWindow(window_name)
                        self.printer.control_led(0, False)
                        return False
                    # Otherwise continue to confirm detection
                    break
                else:
                    # Component not detected
                    cv2.putText(display, f"Waiting for component... ({detection_attempt+1}/{max_detection_attempts})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(display, "Press 'q' to abort", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Draw center crosshair
                    cv2.line(display, (img_center[0]-20, img_center[1]),
                            (img_center[0]+20, img_center[1]), (255, 0, 0), 2)
                    cv2.line(display, (img_center[0], img_center[1]-20),
                            (img_center[0], img_center[1]+20), (255, 0, 0), 2)

                    cv2.imshow(window_name, display)

                    # Check for abort
                    key = cv2.waitKey(500) & 0xFF
                    if key == ord('q') or key == 27:
                        print("  Detection aborted by user")
                        cv2.destroyWindow(window_name)
                        self.printer.control_led(0, False)
                        return False

            detection_attempt += 1
            time.sleep(0.2)

        # Check if detection succeeded
        if center_pos is None or detected_angle is None:
            print("  ERROR: Could not detect component on tool after multiple attempts!")
            cv2.destroyWindow(window_name)
            self.printer.control_led(0, False)
            return False

        # Component detected, continue with orientation correction
        rotation_needed = desired_angle - detected_angle

        # Rotate component to desired angle
        if abs(rotation_needed) > 1:  # Only rotate if difference > 1°
            print(f"  Rotating by {rotation_needed}°...")
            self.printer.rotate_c_axis(rotation_needed)
            self.printer.wait_for_idle()
            print(f"  ✓ Component rotated to {desired_angle}°")

            # CRITICAL: Clear camera buffer after rotation
            # The camera buffer contains old frames from before rotation
            print("  Clearing camera buffer after rotation...")
            #time.sleep(0.1)  # Allow mechanical settling
            for _ in range(5):  # Clear 5 buffered frames
                self.vision_lower.capture_frame()
            time.sleep(0.1)  # One more delay for fresh frame
        else:
            print(f"  ✓ Component already at correct angle (within 1°)")

        # Center component and determine offset (only check desired angle for speed)
        print("  Centering component to determine placement offset...")

        def detect_component_at_desired_angle():
            """
            Detection method that only checks at the desired angle.
            Since we already rotated the component, we only need to find it
            at the current angle, not search through all possible angles.

            Captures from outer scope:
                - lower_template_path: Component template image
                - desired_angle: Target rotation angle
            """
            # Capture fresh frame
            frame = self.vision_lower.capture_frame()
            if frame is None:
                return None, None, None

            # Find component using YOLO or template matching
            if self.use_yolo:
                # YOLO is fast enough to detect at any angle
                result = self.vision_lower.find_component_yolo(expected_angle=desired_angle)
            else:
                # Find component at the desired angle only (no angular search)
                # This is much faster than searching -15 to +16 degrees
                result = self.vision_lower.find_component(
                    lower_template_path,
                    angle=desired_angle,  # Expected angle - component should be at this orientation
                    exact_angle=True  # Only check at this specific angle for speed
                )

            if result[0] is not None:
                pos, angle = result
                return {'X': pos[0], 'Y': pos[1]}, angle, frame
            return None, None, frame

        success, centered_pos = center_target_in_camera(
            printer=self.printer,
            vision=self.vision_lower,
            camera_config=self.camera_configs[0],
            detection_method=detect_component_at_desired_angle,
            tolerance=self.TOLERANCE,
            max_iterations=20,  # Fewer iterations needed since already roughly centered
            feed_rate=150,  # Slower for precision
            debug=False,
            show_display=True
        )

        if success:
            # Calculate offset from camera center
            self.printer.send_gcode_command("M114", check=False)
            current_pos = self.printer.parse_position(self.printer.response)

            # Offset is the difference between current position and camera location
            component_offset = {
                'X': current_pos['X'] - camera_loc[0],
                'Y': current_pos['Y'] - camera_loc[1]
            }
            print(f"  ✓ Component offset: X{component_offset['X']:+.3f}, Y{component_offset['Y']:+.3f}")
        else:
            print("  Warning: Could not center component, using no offset")
            component_offset = {'X': 0, 'Y': 0}

        self.printer.control_led(0, False)

        # 4. Place component (with offset correction)
        print("  4. Placing component...")
        target_pos = placement

        # Method 1: target + component_offset + placement_offset
        final_x = target_pos['x'] + component_offset['X'] + self.placement_offset['x']
        final_y = target_pos['y'] + component_offset['Y'] + self.placement_offset['y']

        # Method 2: current_pos + (target - camera_loc)
        # This should give the same result as Method 1
        alt_final_x = current_pos['X'] + (target_pos['x'] - camera_loc[0])
        alt_final_y = current_pos['Y'] + (target_pos['y'] - camera_loc[1])

        # Sanity check: both methods should match
        diff_x = abs(final_x - alt_final_x)
        diff_y = abs(final_y - alt_final_y)

        # Debug position info (commented out for speed)
        # print(f"  DEBUG: Target: X{target_pos['x']:.3f}, Y{target_pos['y']:.3f} -> Final: X{final_x:.3f}, Y{final_y:.3f}")

        #self.printer.linear_move(z=150,f=6000)  # Safe height
        self.printer.linear_move(x=final_x-2, y=final_y-2, z=target_pos['z']+5,f=6000) #Note - manual offsets
        self.printer.linear_move(x=final_x, y=final_y, f=300)
        self.printer.linear_move(z=target_pos['z']+5,f=6000)
        self.printer.linear_move(z=target_pos['z'])
        self.printer.control_vacuum(False)
        self.printer.control_solenoid(False)  # Release        
        time.sleep(0.1)
        self.printer.linear_move(z=150,f=6000)

        print("  ✓ Component placed!")

        # Return C-axis to home orientation (reverse the rotation we applied)
        if abs(rotation_needed) > 1:
            print(f"  Returning C-axis to home orientation (-{rotation_needed}°)...")
            self.printer.rotate_c_axis(-rotation_needed)
            self.printer.wait_for_idle()

        # 5. Capture verification photo
        # self.capture_verification_photo(
        #     component_type=component['type'],
        #     placement_number=placement_number,
        #     placement_pos=placement
        # )

        return True
