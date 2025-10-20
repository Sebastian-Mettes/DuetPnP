#!/usr/bin/env python3
"""
Unified Tool Calibration Script for DuetPnP

This script calibrates tool offsets for all tools in the system:
- Tools 0, 1, 2: Basic tool calibration using lower camera (camera 0)
- Tool 3: Advanced PnP-based calibration using both cameras

Usage:
    python calibrate_tools.py --tools 0 1 2      # Calibrate tools 0, 1, and 2
    python calibrate_tools.py --tools 3          # Calibrate tool 3 (camera)
    python calibrate_tools.py --tools all        # Calibrate all tools
    python calibrate_tools.py --tool 0           # Calibrate single tool

Options:
    --tools, --tool    Tool number(s) to calibrate (0-3, or 'all')
    --skip-confirm     Skip user confirmation prompts (use with caution)
"""

import sys
import argparse
import time
import json
from machine_vision import VisionTools, CameraConfig, load_camera_config, show_frame_with_overlay
from machine_control import Printer, center_target_in_camera

# Configuration
TARGET_PICKUP_LOCATION = [10.0, 10.0, 1.0]  # For Tool 3 calibration
TARGET_PLACE_LOCATION = [110.0, 110.0, 0.0]  # For Tool 3 calibration
TOOL3_CHECK_HEIGHT = 110.0


def calibrate_basic_tool(printer: Printer, tool_number: int, camera_config: CameraConfig, skip_confirm: bool = False, show_display: bool = True) -> bool:
    """
    Calibrate a basic tool (0, 1, or 2) using the lower camera.

    Args:
        printer: Printer instance
        tool_number: Tool number (0, 1, or 2)
        camera_config: Camera configuration for lower camera
        skip_confirm: Skip user confirmation
        show_display: Show visual feedback window (default: True)

    Returns:
        bool: True if calibration successful
    """
    print("\n" + "="*60)
    print(f"Tool {tool_number} Calibration")
    print("="*60)

    if not skip_confirm:
        input(f"\nPrerequisites for Tool {tool_number}:\n"
              f"  1. Tool {tool_number} is properly installed\n"
              f"  2. Vision parameters calibrated for camera 0 and tool detection\n"
              f"\nPress Enter when ready...")

    try:
        # Get upward camera number from printer configuration
        upward_cam = printer.upward_camera_number

        # Initialize vision tools
        vision = VisionTools(upward_cam, target='tool', tool_number=tool_number, camera_config=camera_config)

        # Select tool and move to camera location
        print(f"\nSelecting Tool {tool_number}...")
        printer.select_tool(tool_number)
        printer.wait_for_idle()
        camera_loc = printer.camera_location
        print(f"Moving to camera location: X{camera_loc[0]:.2f}, Y{camera_loc[1]:.2f}, Z{camera_loc[2]:.2f}")
        printer.linear_move(z=camera_loc[2])
        printer.wait_for_idle()
        printer.linear_move(x=camera_loc[0], y=camera_loc[1])

        printer.control_led(upward_cam, True)  # Turn on upward camera LED
        printer.wait_for_idle()

        # Define detection method for centering
        def detect_tool():
            """Detection callback for centering."""
            frame = vision.capture_frame()
            if frame is None:
                return None, None, None
            tool_pos = vision.find_tool_position()
            if tool_pos is not None:
                x, y = tool_pos
                return {'X': x, 'Y': y}, None, frame
            return None, None, frame

        # Center the tool in camera view
        print(f"\nCentering Tool {tool_number} in camera view...")
        if show_display:
            print("Visual feedback window enabled. Press 'q' to close window at any time.")
        success, final_pos = center_target_in_camera(
            printer=printer,
            vision=vision,
            camera_config=camera_config,
            detection_method=detect_tool,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True,
            show_display=show_display
        )

        if not success:
            print(f"\n✗ Failed to center Tool {tool_number}")
            printer.control_led(0, False)
            vision.cleanup()
            return False

        print(f"✓ Tool {tool_number} centered at machine position: X{final_pos['X']:.2f}, Y{final_pos['Y']:.2f}")

        # Get current position and query existing tool offset
        printer.send_gcode_command("M114", check=False)
        current_pos = printer.parse_position(printer.response)

        try:
            existing_offsets = printer.get_tool_offsets(tool_number)
            existing_offset_x = existing_offsets.get('X', 0.0)
            existing_offset_y = existing_offsets.get('Y', 0.0)
            existing_offset_z = existing_offsets.get('Z', 0.0)
            print(f"\n  Current Tool {tool_number} offset from firmware:")
            print(f"    X: {existing_offset_x:+.3f} mm")
            print(f"    Y: {existing_offset_y:+.3f} mm")
            print(f"    Z: {existing_offset_z:+.3f} mm")
        except Exception as e:
            print(f"  Warning: Could not read existing offset: {e}")
            existing_offset_x = 0.0
            existing_offset_y = 0.0
            existing_offset_z = 0.0

        # Calculate new offset
        # New offset = existing offset - (current_pos - camera_loc)
        correction_x = current_pos['X'] - camera_loc[0]
        correction_y = current_pos['Y'] - camera_loc[1]

        new_offset_x = existing_offset_x - correction_x
        new_offset_y = existing_offset_y - correction_y

        print(f"\n  Measured correction:")
        print(f"    X: {correction_x:+.3f} mm")
        print(f"    Y: {correction_y:+.3f} mm")

        print(f"\n  New offset (for G10 command):")
        print(f"    X: {new_offset_x:+.3f} mm")
        print(f"    Y: {new_offset_y:+.3f} mm")
        print(f"    Z: {existing_offset_z:+.3f} mm (unchanged)")

        # Apply offset
        if not skip_confirm:
            response = input(f"\nApply new offset to Tool {tool_number}? [y/N]: ").strip().lower()
        else:
            response = 'y'

        if response == 'y':
            printer.send_gcode_command(
                f"G10 P{tool_number} X{new_offset_x:.3f} Y{new_offset_y:.3f} Z{existing_offset_z:.3f}",
                check=False
            )
            print(f"✓ Tool {tool_number} offset applied to firmware")
        else:
            print(f"  Offset not applied - you can manually apply it later with:")
            print(f"    G10 P{tool_number} X{new_offset_x:.3f} Y{new_offset_y:.3f} Z{existing_offset_z:.3f}")

        # Save to file
        offset_data = {
            "tool": tool_number,
            "existing_offset_x": round(existing_offset_x, 3),
            "existing_offset_y": round(existing_offset_y, 3),
            "existing_offset_z": round(existing_offset_z, 3),
            "measured_correction_x": round(correction_x, 3),
            "measured_correction_y": round(correction_y, 3),
            "new_offset_x": round(new_offset_x, 3),
            "new_offset_y": round(new_offset_y, 3),
            "calibration_method": "camera centering"
        }

        output_file = f"tool{tool_number}_offset.json"
        with open(output_file, 'w') as f:
            json.dump(offset_data, f, indent=2)
        print(f"✓ Tool {tool_number} offset saved to {output_file}")

        # Cleanup
        printer.control_led(upward_cam, False)
        vision.cleanup()

        print(f"\n✓ Tool {tool_number} calibration complete!")
        return True

    except Exception as e:
        print(f"\n✗ Error during Tool {tool_number} calibration: {e}")
        import traceback
        traceback.print_exc()
        try:
            upward_cam = printer.upward_camera_number
            printer.control_led(upward_cam, False)
            vision.cleanup()
        except:
            pass
        return False


def calibrate_tool3_camera(printer: Printer, cam0_config: CameraConfig, cam2_config: CameraConfig, skip_confirm: bool = False) -> bool:
    """
    Calibrate Tool 3 (downward camera) using PnP method.

    Args:
        printer: Printer instance
        cam0_config: Camera config for lower camera
        cam2_config: Camera config for upper camera
        skip_confirm: Skip user confirmation

    Returns:
        bool: True if calibration successful
    """
    print("\n" + "="*60)
    print("Tool 3 (Downward Camera) Calibration")
    print("="*60)

    if not skip_confirm:
        input("\nPrerequisites:\n"
              "  1. Place circular target object at ~(10, 10) on bed\n"
              "  2. Target should be ~1mm tall\n"
              "  3. Vision parameters calibrated for all cameras/tools\n"
              "\nPress Enter when ready...")

    try:
        # Initialize vision tools
        vision_lower = VisionTools(0, target='target', camera_config=cam0_config)
        vision_upper = VisionTools(2, target='target', camera_config=cam2_config)

        # Step 1: Pick up target with Tool 2
        print("\nStep 1: Picking up target with Tool 2 (PnP tool)")
        print("-" * 60)

        printer.select_tool(2)
        time.sleep(2.5)

        printer.linear_move(x=TARGET_PICKUP_LOCATION[0],
                          y=TARGET_PICKUP_LOCATION[1],
                          z=50)
        time.sleep(1.5)

        printer.control_solenoid(True)
        time.sleep(0.5)
        printer.control_vacuum(True)
        printer.linear_move(z=TARGET_PICKUP_LOCATION[2])
        time.sleep(2.5)

        printer.linear_move(z=150)
        time.sleep(1.5)

        print("✓ Target picked up")

        # Step 2: Center target with lower camera
        print("\nStep 2: Centering target with lower camera")
        print("-" * 60)

        camera_loc = printer.camera_location
        printer.linear_move(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2] + 10)
        time.sleep(1.5)

        printer.control_led(0, True)

        def detect_target(vision):
            """Detection callback for target."""
            frame = vision.capture_frame()
            if frame is None:
                return None, None
            circles = vision.find_circles(frame)
            if circles is not None and len(circles) > 0:
                center_x, center_y = circles[0][:2]
                return {'X': center_x, 'Y': center_y}, None
            return None, None

        success, final_pos = center_target_in_camera(
            printer=printer,
            vision=vision_lower,
            camera_config=cam0_config,
            detection_method=detect_target,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True
        )

        if not success:
            print("\n✗ Failed to center target in lower camera")
            cleanup_tool3(printer, vision_lower, vision_upper)
            return False

        print(f"✓ Target centered at: X{final_pos['X']:.2f}, Y{final_pos['Y']:.2f}")
        printer.control_led(0, False)

        # Step 3: Place target at known location
        print("\nStep 3: Placing target at known location")
        print("-" * 60)

        printer.send_gcode_command("M114", check=False)
        current_pos = printer.parse_position(printer.response)

        offset_x = current_pos['X'] - camera_loc[0]
        offset_y = current_pos['Y'] - camera_loc[1]
        target_x = TARGET_PLACE_LOCATION[0] + offset_x
        target_y = TARGET_PLACE_LOCATION[1] + offset_y

        printer.linear_move(x=target_x, y=target_y)
        time.sleep(1.5)

        printer.linear_move(z=TARGET_PLACE_LOCATION[2])
        printer.wait_for_idle()

        printer.control_solenoid(False)
        time.sleep(0.5)
        printer.control_vacuum(False)
        time.sleep(0.5)

        printer.linear_move(z=150)
        time.sleep(1.5)

        print(f"✓ Target placed at X{target_x:.2f}, Y{target_y:.2f}")

        # Step 4: Find target with Tool 3
        print("\nStep 4: Finding target with Tool 3")
        print("-" * 60)

        printer.select_tool(3)
        time.sleep(2.5)

        printer.linear_move(x=TARGET_PLACE_LOCATION[0],
                          y=TARGET_PLACE_LOCATION[1],
                          z=TOOL3_CHECK_HEIGHT)
        time.sleep(1.5)

        printer.control_led(2, True)

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
            cleanup_tool3(printer, vision_lower, vision_upper)
            return False

        print(f"✓ Target found at: X{final_pos_t3['X']:.2f}, Y{final_pos_t3['Y']:.2f}")
        printer.control_led(2, False)

        # Step 5: Calculate and apply offset
        print("\nStep 5: Calculating Tool 3 offset")
        print("-" * 60)

        try:
            existing_offsets = printer.get_tool_offsets(3)
            existing_offset_x = existing_offsets.get('X', 0.0)
            existing_offset_y = existing_offsets.get('Y', 0.0)
            existing_offset_z = existing_offsets.get('Z', 0.0)
            print(f"  Current Tool 3 offset: X{existing_offset_x:+.3f}, Y{existing_offset_y:+.3f}, Z{existing_offset_z:+.3f}")
        except:
            existing_offset_x = 0.0
            existing_offset_y = 0.0
            existing_offset_z = 0.0

        correction_x = final_pos_t3['X'] - TARGET_PLACE_LOCATION[0]
        correction_y = final_pos_t3['Y'] - TARGET_PLACE_LOCATION[1]

        new_offset_x = existing_offset_x - correction_x
        new_offset_y = existing_offset_y - correction_y

        print(f"  Measured correction: X{correction_x:+.3f}, Y{correction_y:+.3f}")
        print(f"  New offset: X{new_offset_x:+.3f}, Y{new_offset_y:+.3f}, Z{existing_offset_z:+.3f}")

        if not skip_confirm:
            response = input("\nApply new offset to Tool 3? [y/N]: ").strip().lower()
        else:
            response = 'y'

        if response == 'y':
            printer.send_gcode_command(
                f"G10 P3 X{new_offset_x:.3f} Y{new_offset_y:.3f} Z{existing_offset_z:.3f}",
                check=False
            )
            print("✓ Tool 3 offset applied to firmware")

        # Save to file
        offset_data = {
            "tool": 3,
            "existing_offset_x": round(existing_offset_x, 3),
            "existing_offset_y": round(existing_offset_y, 3),
            "existing_offset_z": round(existing_offset_z, 3),
            "measured_correction_x": round(correction_x, 3),
            "measured_correction_y": round(correction_y, 3),
            "new_offset_x": round(new_offset_x, 3),
            "new_offset_y": round(new_offset_y, 3),
            "calibration_method": "PnP target placement"
        }

        with open("tool3_offset.json", 'w') as f:
            json.dump(offset_data, f, indent=2)

        cleanup_tool3(printer, vision_lower, vision_upper)

        print("\n✓ Tool 3 calibration complete!")
        return True

    except Exception as e:
        print(f"\n✗ Error during Tool 3 calibration: {e}")
        import traceback
        traceback.print_exc()
        try:
            cleanup_tool3(printer, vision_lower, vision_upper)
        except:
            pass
        return False


def cleanup_tool3(printer, vision_lower, vision_upper):
    """Clean up after Tool 3 calibration."""
    try:
        printer.control_led(0, False)
        printer.control_led(2, False)
        printer.control_vacuum(False)
        printer.control_solenoid(False)
        vision_lower.cleanup()
        vision_upper.cleanup()
    except:
        pass


def main():
    """Main entry point for tool calibration."""
    parser = argparse.ArgumentParser(
        description='Calibrate tool offsets for DuetPnP system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--tools', '--tool', nargs='+', required=True,
                       help='Tool number(s) to calibrate (0-3, or "all")')
    parser.add_argument('--skip-confirm', action='store_true',
                       help='Skip user confirmation prompts')

    args = parser.parse_args()

    # Parse tool numbers
    if 'all' in args.tools:
        tool_numbers = [0, 1, 2, 3]
    else:
        tool_numbers = []
        for t in args.tools:
            try:
                tool_num = int(t)
                if tool_num < 0 or tool_num > 3:
                    print(f"Error: Tool number must be 0-3, got {tool_num}")
                    return 1
                tool_numbers.append(tool_num)
            except ValueError:
                print(f"Error: Invalid tool number '{t}'")
                return 1

    tool_numbers = sorted(set(tool_numbers))  # Remove duplicates and sort

    print("="*60)
    print("DuetPnP Tool Calibration")
    print("="*60)
    print(f"Tools to calibrate: {', '.join(str(t) for t in tool_numbers)}\n")

    try:
        # Initialize hardware
        print("Initializing hardware...")
        printer = Printer(upward_camera_number=0, debug=False)

        # Load camera configurations
        cam0_config = load_camera_config(0)  # Lower/upward camera
        cam2_config = load_camera_config(2)  # Upper/downward camera

        print("✓ Hardware initialized\n")

        # Calibrate each tool
        results = {}
        for tool_num in tool_numbers:
            if tool_num in [0, 1, 2]:
                results[tool_num] = calibrate_basic_tool(printer, tool_num, cam0_config, args.skip_confirm)
            elif tool_num == 3:
                results[tool_num] = calibrate_tool3_camera(printer, cam0_config, cam2_config, args.skip_confirm)

        # Summary
        print("\n" + "="*60)
        print("Calibration Summary")
        print("="*60)
        for tool_num, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"  Tool {tool_num}: {status}")
        print("="*60)

        # Cleanup
        printer.close()

        all_success = all(results.values())
        return 0 if all_success else 1

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")
        try:
            printer.close()
        except:
            pass
        return 1

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            printer.close()
        except:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
