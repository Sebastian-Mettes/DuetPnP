"""
PnP Operations Module for DuetPnP

This module handles PnP-specific operations including:
- ConfigManager: Loads placement configs, templates, camera offsets
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


class ConfigManager:
    """
    Manages all configuration files for PnP operations.
    
    Handles:
    - Placement configuration (placement_config.json)
    - Component templates (upper and lower)
    - Camera offset calibration
    """
    
    def __init__(self, config_file: str = "placement_config.json"):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to placement configuration JSON
        """
        self.config_file = config_file
        self.config = None
        self.upper_templates = {}
        self.lower_templates = {}
        self.camera_offset = {'X': 0, 'Y': 0, 'Z': 0}
        
        self.load_config()
        self.load_camera_offset()
    
    def load_config(self):
        """
        Load placement configuration from JSON file.
        
        Expected format:
        {
            "components": [
                {
                    "type": "component_name",
                    "upper_template": "path/to/above_view.png",
                    "lower_template": "path/to/below_view.png",
                    "feed_number": 0,
                    "reel_location": {"x": 0, "y": 0, "z": 0},
                    "reel_focus": 101.25,
                    "placements": [
                        {"x": 0, "y": 0, "z": 0, "rotation": 0}
                    ]
                }
            ],
            "vacuum_pin": "fan1",
            "solenoid_pin": "fan2"
        }
        """
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            
            # Load templates for each component
            for component in self.config['components']:
                comp_type = component['type']
                
                # Load upper template (feeder view)
                upper_path = component['upper_template']
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
    
    def load_camera_offset(self):
        """Load camera offset from calibration file."""
        try:
            with open('camera_offset.json', 'r') as f:
                self.camera_offset = json.load(f)
            print(f"Loaded camera offset: {self.camera_offset}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load camera offset: {e}")
            self.camera_offset = {'X': 0, 'Y': 0, 'Z': 0}
    
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
                 camera_configs: Dict[int, CameraConfig]):
        """
        Initialize PnP workflow.
        
        Args:
            printer: Printer instance
            feeder: Feeder instance
            config_manager: ConfigManager instance
            camera_configs: Dictionary mapping camera numbers to CameraConfig instances
        """
        self.printer = printer
        self.feeder = feeder
        self.config = config_manager
        self.camera_configs = camera_configs
        
        # Initialize vision tools for each camera
        self.vision_upper = VisionTools(2, target='tool', 
                                       camera_config=camera_configs[2])
        self.vision_lower = VisionTools(0, target='tool',
                                       camera_config=camera_configs[0])
        
        # Centering parameters
        self.TOLERANCE = 2
        self.MAX_ITERATIONS = 20

        # Checkpoint state
        self.checkpoint_file = "pnp_checkpoint.json"
        self.current_component_index = 0
        self.current_placement_index = 0
        self.completed_placements = []
        self.failed_placements = []

        print("PnP Workflow initialized")

    def save_checkpoint(self):
        """Save current progress to checkpoint file."""
        from datetime import datetime
        checkpoint_data = {
            "description": "PnP workflow checkpoint for resume capability",
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "status": "paused",
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

            self.current_component_index = checkpoint_data.get('current_component_index', 0)
            self.current_placement_index = checkpoint_data.get('current_placement_index', 0)
            self.completed_placements = checkpoint_data.get('completed_placements', [])
            self.failed_placements = checkpoint_data.get('failed_placements', [])

            print(f"✓ Checkpoint loaded from {self.checkpoint_file}")
            print(f"  Resuming from component {self.current_component_index}, placement {self.current_placement_index}")
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
                    success = self.place_component(component, placement)

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
    
    def place_component(self, component: Dict, placement: Dict) -> bool:
        """
        Pick component from feeder and place at target location.
        
        Args:
            component: Component configuration dictionary
            placement: Placement location dictionary
        
        Returns:
            True if successful, False otherwise
        """
        # 1. Locate component in feeder
        print("  1. Locating component...")
        reel_loc = component['reel_location']
        reel_focus = component.get('reel_focus', 101.25)
        
        # Move upper camera to feeder location
        self.printer.select_tool(3)  # Camera tool
        self.printer.control_led(2, True)  # Upper camera LED
        self.printer.linear_move(z=150)  # Safe height
        self.printer.linear_move(x=reel_loc['x'], y=reel_loc['y'], z=reel_focus)
        time.sleep(1)
        
        # Center component in view
        template_path = component['upper_template']
        detection_fn = lambda: self.vision_upper.find_component(template_path)
        
        success, final_pos = center_target_in_camera(
            self.printer,
            self.vision_upper,
            self.camera_configs[2],
            detection_fn,
            tolerance=self.TOLERANCE,
            max_iterations=self.MAX_ITERATIONS
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
        self.printer.select_tool(2)  # PnP tool

        self.printer.control_vacuum(True)
        self.printer.linear_move(x=pickup_pos['X'], y=pickup_pos['Y'], z=reel_loc['z'])
        time.sleep(0.5)
        self.printer.control_solenoid(True)
        time.sleep(0.5)
        self.printer.linear_move(z=150)  # Lift
        
        # 3. Determine orientation with lower camera
        print("  3. Checking component orientation...")
        self.printer.control_led(0, True)  # Lower camera LED
        camera_loc = self.printer.camera_location
        self.printer.linear_move(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2])
        time.sleep(1)
        
        # Detect component and rotation
        lower_template_path = component['lower_template']
        detection_result = self.vision_lower.find_component(lower_template_path)
        
        if detection_result[0] is None:
            print("  Warning: Could not detect component on tool!")
            self.printer.control_led(0, False)
            # Continue anyway - place at 0° rotation
        else:
            center_pos, detected_angle = detection_result
            desired_angle = placement.get('rotation', 0)
            rotation_needed = desired_angle - detected_angle
            print(f"  Component detected at {detected_angle}°, target is {desired_angle}°")

            # Rotate component to desired angle
            if abs(rotation_needed) > 1:  # Only rotate if difference > 1°
                print(f"  Rotating by {rotation_needed}°...")
                self.printer.rotate_c_axis(rotation_needed)
                self.printer.wait_for_idle()
                print(f"  ✓ Component rotated to {desired_angle}°")
            else:
                print(f"  ✓ Component already at correct angle (within 1°)")

        self.printer.control_led(0, False)
        
        # 4. Place component
        print("  4. Placing component...")
        target_pos = placement
        self.printer.linear_move(z=150)  # Safe height
        self.printer.linear_move(x=target_pos['x'], y=target_pos['y'])
        self.printer.linear_move(z=target_pos['z'])
        self.printer.wait_for_idle()  # CRITICAL: Wait for Z to reach placement height

        self.printer.control_solenoid(False)  # Release
        time.sleep(0.5)
        self.printer.control_vacuum(False)
        time.sleep(0.5)
        self.printer.linear_move(z=150)
        
        print("  ✓ Component placed!")
        return True
