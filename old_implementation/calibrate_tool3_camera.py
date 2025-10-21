#!/usr/bin/env python3
"""
Tool 3 (Downward Camera) Calibration Script

This script calibrates the downward-facing camera (Tool 3) by:
1. Using Tool 2 (PnP tool) to pick up a target object from the bed
2. Centering the target using the lower camera (camera 0) to determine offset
3. Placing the target at a known location (110, 110)
4. Using Tool 3 (camera tool) to find the target at the known location
5. Calculating Tool 3's offset relative to machine coordinates

Usage:
    python calibrate_tool3_camera.py

Requirements:
    - Target object (circular) placed at approximately (10, 10) on bed
    - Target should be ~1mm tall
    - Vision parameters calibrated for target detection:
      * vision_params_camera0_target.json (run: python generate_tool_vision_params.py 0 target)
      * vision_params_camera2_target.json (run: python generate_tool_vision_params.py 2 target)
"""

import time
import json
from machine_vision import VisionTools, CameraConfig, load_camera_config, show_frame_with_overlay
from machine_control import Printer, center_target_in_camera

# Configuration
TARGET_PICKUP_LOCATION = [10.0, 10.0, 1.0]  # X, Y, Z where target sits on bed
TARGET_PLACE_LOCATION = [110.0, 110.0, 0.0]  # Known placement location
TOOL3_CHECK_HEIGHT = 158.0  # Z height for Tool 3 to check target


def main():
    """Main calibration workflow for Tool 3 (downward camera)."""
    print("="*60)
    print("Tool 3 (Downward Camera) Calibration")
    print("="*60)
    print("\nThis will calibrate the downward-facing camera (Tool 3)")
    print("by placing a target at a known location and measuring offset.\n")

    input("Prerequisites:\n"
          "  1. Place circular target object at ~(10, 10) on bed\n"
          "  2. Target should be ~1mm tall\n"
          "  3. Vision parameters calibrated for all cameras/tools\n"
          "\nPress Enter when ready...")

    try:
        # Initialize hardware
        print("\nInitializing hardware...")
        printer = Printer(upward_camera_number=0, debug=False)

        # Load camera configurations
        cam0_config = load_camera_config(0)  # Lower/upward camera
        cam2_config = load_camera_config(2)  # Upper/downward camera

        # Initialize vision tools
        vision_lower = VisionTools(0, target='target', camera_config=cam0_config)
        vision_upper = VisionTools(2, target='target', camera_config=cam2_config)

        print("✓ Hardware initialized\n")

        # Step 1: Pick up target with Tool 2
        print("Step 1: Picking up target with Tool 2 (PnP tool)")
        print("-" * 60)

        printer.select_tool(2)
        time.sleep(2.5)

        # Move to target location
        printer.linear_move(x=TARGET_PICKUP_LOCATION[0],
                          y=TARGET_PICKUP_LOCATION[1],
                          z=50)
        time.sleep(1.5)

        # Activate vacuum and pick up
        printer.control_solenoid(True)
        time.sleep(0.5)
        printer.control_vacuum(True)
        printer.linear_move(z=TARGET_PICKUP_LOCATION[2])
        time.sleep(2.5)

        # Lift target
        printer.linear_move(z=150)
        time.sleep(1.5)

        print("✓ Target picked up\n")

        # Step 2: Center target with lower camera to determine offset
        print("Step 2: Centering target with lower camera")
        print("-" * 60)

        # Move to camera location
        camera_loc = printer.camera_location
        printer.linear_move(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2] + 10)
        time.sleep(1.5)

        printer.control_led(0, True)  # Turn on lower camera LED

        # Center the target using generic centering function
        def detect_target_lower():
            """Detection callback for target on lower camera."""
            frame = vision_lower.capture_frame()
            if frame is None:
                return None, None, None
            # Use find_tool_position for target detection (uses vision_params_camera0_target.json)
            target_pos = vision_lower.find_tool_position()
            if target_pos is not None:
                x, y = target_pos
                return {'X': x, 'Y': y}, None, frame
            return None, None, frame

        print("  Centering target in lower camera view...")
        print("  Visual feedback window will show camera view. Press 'q' to abort.")
        success, final_pos = center_target_in_camera(
            printer=printer,
            vision=vision_lower,
            camera_config=cam0_config,
            detection_method=detect_target_lower,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True,
            show_display=True
        )

        if not success:
            print("\n✗ Failed to center target in lower camera")
            cleanup(printer, vision_lower, vision_upper)
            return 1

        print(f"  ✓ Target centered at machine position: X{final_pos['X']:.2f}, Y{final_pos['Y']:.2f}")

        printer.control_led(0, False)

        # Step 3: Place target at known location
        print("\nStep 3: Placing target at known location")
        print("-" * 60)

        # Get current position to calculate placement offset
        printer.send_gcode_command("M114", check=False)
        current_pos = printer.parse_position(printer.response)

        # Calculate target placement position with offset
        offset_x = current_pos['X'] - camera_loc[0]
        offset_y = current_pos['Y'] - camera_loc[1]
        target_x = TARGET_PLACE_LOCATION[0] + offset_x
        target_y = TARGET_PLACE_LOCATION[1] + offset_y

        print(f"  Placing at: X{target_x:.2f}, Y{target_y:.2f}")

        # Move to placement location
        printer.linear_move(x=target_x, y=target_y)
        time.sleep(1.5)

        # Lower and release
        printer.linear_move(z=TARGET_PLACE_LOCATION[2])
        printer.wait_for_idle()

        printer.control_solenoid(False)
        time.sleep(0.5)
        printer.control_vacuum(False)
        time.sleep(0.5)

        # Lift away
        printer.linear_move(z=150)
        time.sleep(1.5)

        print(f"✓ Target placed at X{target_x:.2f}, Y{target_y:.2f}\n")

        # Step 4: Find target with Tool 3 (upper camera)
        print("Step 4: Finding target with Tool 3 (downward camera)")
        print("-" * 60)

        printer.select_tool(3)
        time.sleep(2.5)

        # Move Tool 3 to where we think the target is
        printer.linear_move(x=TARGET_PLACE_LOCATION[0],
                          y=TARGET_PLACE_LOCATION[1],
                          z=TOOL3_CHECK_HEIGHT)
        time.sleep(1.5)

        printer.control_led(2, True)  # Turn on upper camera LED

        # Center the target using Tool 3
        def detect_target_upper():
            """Detection callback for target on upper camera."""
            frame = vision_upper.capture_frame()
            if frame is None:
                return None, None, None
            # Use find_tool_position for target detection (uses vision_params_camera2_target.json)
            target_pos = vision_upper.find_tool_position()
            if target_pos is not None:
                x, y = target_pos
                return {'X': x, 'Y': y}, None, frame
            return None, None, frame

        print("  Centering target in Tool 3 camera view...")
        print("  Visual feedback window will show camera view. Press 'q' to abort.")
        success, final_pos_t3 = center_target_in_camera(
            printer=printer,
            vision=vision_upper,
            camera_config=cam2_config,
            detection_method=detect_target_upper,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True,
            show_display=True
        )

        if not success:
            print("\n✗ Failed to find target with Tool 3")
            cleanup(printer, vision_lower, vision_upper)
            return 1

        print(f"  ✓ Target found at machine position: X{final_pos_t3['X']:.2f}, Y{final_pos_t3['Y']:.2f}")

        printer.control_led(2, False)

        # Step 5: Calculate and save Tool 3 offset
        print("\nStep 5: Calculating Tool 3 offset")
        print("-" * 60)

        # Get axis mappings for Tool 3 (e.g., X->U, Y->V for some tools)
        printer.send_gcode_command("M563 P3", check=False)
        axis_maps = printer.parse_axis_mapping(printer.response)
        print(f"\n  Axis mappings for Tool 3: {axis_maps}")

        # Get current tool offsets from firmware
        printer.send_gcode_command("G10 P3", check=False)
        current_offsets = printer.parse_tool_offsets(printer.response, 3)

        print(f"\n  Current offsets from firmware:")
        for axis, value in sorted(current_offsets.items()):
            print(f"    {axis}: {value:+.3f} mm")

        # Calculate new offsets using mapped axes (same approach as calibrate_tools.py)
        # Expected position is where we placed the target
        expected_position = [TARGET_PLACE_LOCATION[0], TARGET_PLACE_LOCATION[1], TOOL3_CHECK_HEIGHT]  # [X, Y, Z]
        new_offsets = {}

        for source_axis, mapped_axis in axis_maps.items():
            # Get the index for X, Y, or Z (0, 1, or 2)
            axis_index = 'XYZ'.index(source_axis)
            # Calculate the difference in the source axis (X, Y, or Z)
            source_diff = final_pos_t3[source_axis] - expected_position[axis_index]
            # Add this difference to the current offset of the mapped axis
            new_offsets[mapped_axis] = current_offsets[mapped_axis] - source_diff

        print(f"\n  Calculated position differences (source coordinates):")
        print(f"    X: {final_pos_t3['X'] - expected_position[0]:+.3f} mm")
        print(f"    Y: {final_pos_t3['Y'] - expected_position[1]:+.3f} mm")

        print(f"\n  New offsets (mapped to physical axes):")
        for axis, value in sorted(new_offsets.items()):
            print(f"    {axis}: {value:+.3f} mm")

        # Apply offset
        response = input("\nApply new offset to Tool 3? [y/N]: ").strip().lower()
        if response == 'y':
            # Build G10 command with mapped axes
            offset_params = ' '.join(f"{axis}{value:.3f}" for axis, value in sorted(new_offsets.items()))
            printer.send_gcode_command(
                f"G10 P3 {offset_params}",
                check=False
            )
            print(f"✓ Tool 3 offset applied to firmware")
        else:
            offset_params = ' '.join(f"{axis}{value:.3f}" for axis, value in sorted(new_offsets.items()))
            print(f"  Offset not applied - you can manually apply it later with:")
            print(f"    G10 P3 {offset_params}")

        # Save to file with axis mapping information
        offset_data = {
            "tool": 3,
            "axis_mappings": axis_maps,
            "current_offsets": {k: round(v, 3) for k, v in current_offsets.items()},
            "new_offsets": {k: round(v, 3) for k, v in new_offsets.items()},
            "position_difference_X": round(final_pos_t3['X'] - expected_position[0], 3),
            "position_difference_Y": round(final_pos_t3['Y'] - expected_position[1], 3),
            "expected_position": expected_position,
            "actual_position": [final_pos_t3['X'], final_pos_t3['Y'], final_pos_t3.get('Z', TOOL3_CHECK_HEIGHT)],
            "calibration_method": "PnP target placement with axis mapping"
        }

        output_file = "tool3_offset.json"
        with open(output_file, 'w') as f:
            json.dump(offset_data, f, indent=2)
        print(f"✓ Tool 3 offset saved to {output_file}")

        print("\n" + "="*60)
        print("✓ Tool 3 calibration complete!")
        print("="*60)

        cleanup(printer, vision_lower, vision_upper)
        return 0

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
        try:
            cleanup(printer, vision_lower, vision_upper)
        except:
            pass
        return 1

    except Exception as e:
        print(f"\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        try:
            cleanup(printer, vision_lower, vision_upper)
        except:
            pass
        return 1


def cleanup(printer, vision_lower, vision_upper):
    """Clean up resources."""
    try:
        printer.control_led(0, False)
        printer.control_led(2, False)
        printer.control_vacuum(False)
        printer.control_solenoid(False)
        vision_lower.cleanup()
        vision_upper.cleanup()
        printer.close()
        print("\n✓ Cleanup complete")
    except Exception as e:
        print(f"Warning during cleanup: {e}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
