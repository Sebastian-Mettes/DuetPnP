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
    - All vision parameters already calibrated for cameras and tools
"""

import time
import json
from machine_vision import VisionTools, CameraConfig, load_camera_config, show_frame_with_overlay
from machine_control import Printer, center_target_in_camera

# Configuration
TARGET_PICKUP_LOCATION = [10.0, 10.0, 1.0]  # X, Y, Z where target sits on bed
TARGET_PLACE_LOCATION = [110.0, 110.0, 0.0]  # Known placement location
TOOL3_CHECK_HEIGHT = 110.0  # Z height for Tool 3 to check target


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
        def detect_target(vision):
            """Detection callback for centering."""
            frame = vision.capture_frame()
            if frame is None:
                return None, None
            # Use find_circles for target detection
            circles = vision.find_circles(frame)
            if circles is not None and len(circles) > 0:
                center_x, center_y = circles[0][:2]
                return {'X': center_x, 'Y': center_y}, None
            return None, None

        print("  Centering target in lower camera view...")
        success, final_pos = center_target_in_camera(
            printer=printer,
            vision=vision_lower,
            camera_config=cam0_config,
            detection_method=detect_target(vision_lower),
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True
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
        print("  Centering target in Tool 3 camera view...")
        success, final_pos_t3 = center_target_in_camera(
            printer=printer,
            vision=vision_upper,
            camera_config=cam2_config,
            detection_method=detect_target,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True
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

        # Query existing tool offset from firmware
        try:
            existing_offsets = printer.get_tool_offsets(3)
            existing_offset_x = existing_offsets.get('X', 0.0)
            existing_offset_y = existing_offsets.get('Y', 0.0)
            print(f"  Current Tool 3 offset from firmware (G10 P3):")
            print(f"    X: {existing_offset_x:+.3f} mm")
            print(f"    Y: {existing_offset_y:+.3f} mm")
        except Exception as e:
            print(f"  Warning: Could not read existing offset from firmware: {e}")
            print(f"  Using zero as existing offset")
            existing_offset_x = 0.0
            existing_offset_y = 0.0

        # Tool 3 measured correction is the difference between where we think we are
        # (TARGET_PLACE_LOCATION) and where we actually are (final_pos_t3)
        correction_x = final_pos_t3['X'] - TARGET_PLACE_LOCATION[0]
        correction_y = final_pos_t3['Y'] - TARGET_PLACE_LOCATION[1]

        print(f"\n  Measured correction from calibration:")
        print(f"    X: {correction_x:+.3f} mm")
        print(f"    Y: {correction_y:+.3f} mm")

        # New offset = existing offset - correction
        # (We subtract because if we're measuring +0.5mm too far, we need -0.5mm offset)
        new_offset_x = existing_offset_x - correction_x
        new_offset_y = existing_offset_y - correction_y

        print(f"\n  New offset (for G10 command):")
        print(f"    X: {new_offset_x:+.3f} mm")
        print(f"    Y: {new_offset_y:+.3f} mm")

        # Save to file
        offset_data = {
            "tool": 3,
            "description": "Tool 3 (downward camera) offset calibration results",
            "existing_offset_x": round(existing_offset_x, 3),
            "existing_offset_y": round(existing_offset_y, 3),
            "measured_correction_x": round(correction_x, 3),
            "measured_correction_y": round(correction_y, 3),
            "new_offset_x": round(new_offset_x, 3),
            "new_offset_y": round(new_offset_y, 3),
            "calibration_method": "PnP target placement",
            "target_location": TARGET_PLACE_LOCATION,
            "measured_location": [final_pos_t3['X'], final_pos_t3['Y']]
        }

        output_file = "tool3_offset.json"
        with open(output_file, 'w') as f:
            json.dump(offset_data, f, indent=2)

        print(f"\n✓ Tool 3 offset saved to {output_file}")

        # Update tool offset in RRF (optional)
        response = input("\nApply new offset to Tool 3 in firmware? [y/N]: ").strip().lower()
        if response == 'y':
            # Send G10 command to set tool offset
            printer.send_gcode_command(f"G10 P3 X{new_offset_x:.3f} Y{new_offset_y:.3f}", check=False)
            print("✓ Tool offset applied to firmware")
            print(f"  Command sent: G10 P3 X{new_offset_x:.3f} Y{new_offset_y:.3f}")
        else:
            print("  Offset not applied - you can manually apply it later with:")
            print(f"    G10 P3 X{new_offset_x:.3f} Y{new_offset_y:.3f}")

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
