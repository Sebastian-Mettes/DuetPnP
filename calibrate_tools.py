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
from machine_vision import VisionTools, CameraConfig, load_camera_config
from machine_control import Printer, center_target_in_camera

# Configuration
TARGET_PICKUP_LOCATION = [10.0, 10.0, -2]  # For Tool 3 calibration
TARGET_PLACE_LOCATION = [110.0, 50.0, 0]  # For Tool 3 calibration
TOOL3_CHECK_HEIGHT = 158.0


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

        # Clear camera buffer after movement and LED turn-on
        time.sleep(0.3)  # Allow LED to stabilize
        for _ in range(5):
            vision.capture_frame()
        time.sleep(0.1)

        # Show live preview before centering starts
        if show_display:
            import cv2
            print("\nShowing camera preview...")
            print("Press 's' to start centering, or 'q' to quit")
            print(f"[DEBUG] OpenCV version: {cv2.__version__}")

            # Create window ONCE before loop (like generate_tool_vision_params.py)
            window_name = "Camera Preview"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            preview_running = True
            frame_count = 0
            while preview_running:
                frame = vision.capture_frame()
                frame_count += 1

                if frame is not None:
                    if frame_count == 1:
                        print(f"[DEBUG] First frame captured: shape={frame.shape}, dtype={frame.dtype}")

                    tool_pos = vision.find_tool_position()
                    img_center = vision.get_image_center()

                    # Draw directly like generate_tool_vision_params.py
                    display = frame.copy()

                    # Draw center crosshair
                    cx, cy = img_center
                    cv2.line(display, (cx-20, cy), (cx+20, cy), (255, 0, 0), 2)
                    cv2.line(display, (cx, cy-20), (cx, cy+20), (255, 0, 0), 2)

                    if tool_pos is not None:
                        print(f"[DEBUG] Frame {frame_count}: Tool detected at {tool_pos}")
                        # Draw detected tool
                        x, y = int(tool_pos[0]), int(tool_pos[1])
                        cv2.circle(display, (x, y), 20, (0, 255, 0), 3)
                        cv2.circle(display, (x, y), 2, (0, 0, 255), -1)
                        cv2.putText(display, "Tool Detected - Press 's' to start", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        if frame_count % 10 == 1:  # Print every 10 frames
                            print(f"[DEBUG] Frame {frame_count}: No tool detected")
                        cv2.putText(display, "No Tool Detected - Press 's' or 'q'", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Show frame directly (like generate_tool_vision_params.py)
                    cv2.imshow(window_name, display)

                    if frame_count == 1:
                        print("[DEBUG] cv2.imshow called directly, cv2.waitKey about to be called")

                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('s'):
                        print("[DEBUG] 's' key pressed")
                        preview_running = False
                        cv2.destroyWindow("Camera Preview")
                    elif key == ord('q'):
                        print("[DEBUG] 'q' key pressed")
                        print("\nCalibration cancelled by user")
                        printer.control_led(upward_cam, False)
                        vision.cleanup()
                        return False
                    elif key != 255:  # 255 means no key pressed
                        print(f"[DEBUG] Key pressed: {key}")
                else:
                    print(f"[DEBUG] Frame {frame_count}: Failed to capture frame")

                time.sleep(0.05)

        # Define detection method for centering
        debug_mode = True  # Enable debug for centering
        def detect_tool():
            """Detection callback for centering."""
            frame = vision.capture_frame()
            if frame is None:
                if debug_mode:
                    print("  [detect_tool] Failed to capture frame")
                return None, None, None
            tool_pos = vision.find_tool_position()
            if tool_pos is not None:
                x, y = tool_pos
                if debug_mode:
                    print(f"  [detect_tool] Tool detected at ({x}, {y}), frame shape: {frame.shape}")
                return {'X': x, 'Y': y}, None, frame
            if debug_mode:
                print(f"  [detect_tool] No tool detected, frame shape: {frame.shape}")
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
            debug=debug_mode,
            show_display=show_display
        )

        if not success:
            print(f"\n✗ Failed to center Tool {tool_number}")
            printer.control_led(0, False)
            vision.cleanup()
            return False

        print(f"✓ Tool {tool_number} centered at machine position: X{final_pos['X']:.2f}, Y{final_pos['Y']:.2f}")

        # Get axis mappings for this tool (e.g., X->U, Y->V for some tools)
        printer.send_gcode_command(f"M563 P{tool_number}", check=False)
        axis_maps = printer.parse_axis_mapping(printer.response)
        print(f"\n  Axis mappings for Tool {tool_number}: {axis_maps}")

        # Get current position
        printer.send_gcode_command("M114", check=False)
        current_pos = printer.parse_position(printer.response)

        # Get current tool offsets
        printer.send_gcode_command(f"G10 P{tool_number}", check=False)
        current_offsets = printer.parse_tool_offsets(printer.response, tool_number)

        print(f"\n  Current offsets from firmware:")
        for axis, value in sorted(current_offsets.items()):
            print(f"    {axis}: {value:+.3f} mm")

        # Calculate new offsets using mapped axes (like old calibration.py)
        # For each source axis (X, Y, Z), calculate difference and apply to mapped axis
        expected_position = [camera_loc[0], camera_loc[1], camera_loc[2]]  # [X, Y, Z]
        new_offsets = {}

        for source_axis, mapped_axis in axis_maps.items():
            # Get the index for X, Y, or Z (0, 1, or 2)
            axis_index = 'XYZ'.index(source_axis)
            # Calculate the difference in the source axis (X, Y, or Z)
            source_diff = current_pos[source_axis] - expected_position[axis_index]
            # Add this difference to the current offset of the mapped axis
            new_offsets[mapped_axis] = current_offsets[mapped_axis] - source_diff

        print(f"\n  Calculated position differences (source coordinates):")
        print(f"    X: {current_pos['X'] - expected_position[0]:+.3f} mm")
        print(f"    Y: {current_pos['Y'] - expected_position[1]:+.3f} mm")

        print(f"\n  New offsets (mapped to physical axes):")
        for axis, value in sorted(new_offsets.items()):
            print(f"    {axis}: {value:+.3f} mm")

        # Apply offset
        if not skip_confirm:
            response = input(f"\nApply new offset to Tool {tool_number}? [y/N]: ").strip().lower()
        else:
            response = 'y'

        if response == 'y':
            # Build G10 command with mapped axes
            offset_params = ' '.join(f"{axis}{value:.3f}" for axis, value in sorted(new_offsets.items()))
            printer.send_gcode_command(
                f"G10 P{tool_number} {offset_params}",
                check=False
            )
            print(f"✓ Tool {tool_number} offset applied to firmware")
        else:
            offset_params = ' '.join(f"{axis}{value:.3f}" for axis, value in sorted(new_offsets.items()))
            print(f"  Offset not applied - you can manually apply it later with:")
            print(f"    G10 P{tool_number} {offset_params}")

        # Save to file with axis mapping information
        offset_data = {
            "tool": tool_number,
            "axis_mappings": axis_maps,
            "current_offsets": {k: round(v, 3) for k, v in current_offsets.items()},
            "new_offsets": {k: round(v, 3) for k, v in new_offsets.items()},
            "position_difference_X": round(current_pos['X'] - expected_position[0], 3),
            "position_difference_Y": round(current_pos['Y'] - expected_position[1], 3),
            "expected_position": expected_position,
            "actual_position": [current_pos['X'], current_pos['Y'], current_pos['Z']],
            "calibration_method": "camera centering with axis mapping"
        }

        output_file = f"config/tool{tool_number}_offset.json"
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


def calibrate_tool3_camera(printer: Printer, cam0_config: CameraConfig, cam2_config: CameraConfig, skip_confirm: bool = False, show_display: bool = True) -> bool:
    """
    Calibrate Tool 3 (downward camera) using PnP method.

    Args:
        printer: Printer instance
        cam0_config: Camera config for lower camera
        cam2_config: Camera config for upper camera
        skip_confirm: Skip user confirmation
        show_display: Show visual feedback window (default: True)

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
        printer.wait_for_idle()

        printer.linear_move(x=TARGET_PICKUP_LOCATION[0],
                          y=TARGET_PICKUP_LOCATION[1],
                          z=50)
        printer.wait_for_idle()

        printer.control_solenoid(True)
        time.sleep(0.25)
        printer.control_vacuum(True)
        printer.linear_move(z=TARGET_PICKUP_LOCATION[2])
        printer.wait_for_idle()

        printer.linear_move(z=150)
        printer.wait_for_idle()

        print("✓ Target picked up")

        # Step 2: Center target with lower camera
        print("\nStep 2: Centering target with lower camera")
        print("-" * 60)

        camera_loc = printer.camera_location
        printer.linear_move(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2])
        printer.wait_for_idle()

        printer.control_led(0, True)
        time.sleep(0.3)  # Allow LED to stabilize

        # Clear camera buffer after movement and LED turn-on
        for _ in range(5):
            vision_lower.capture_frame()
        time.sleep(0.1)

        def detect_target_lower():
            """Detection callback for target on lower camera."""
            frame = vision_lower.capture_frame()
            if frame is None:
                return None, None, None
            target_pos = vision_lower.find_tool_position()  # Uses target vision params
            if target_pos is not None:
                x, y = target_pos
                return {'X': x, 'Y': y}, None, frame
            return None, None, frame

        success, final_pos = center_target_in_camera(
            printer=printer,
            vision=vision_lower,
            camera_config=cam0_config,
            detection_method=detect_target_lower,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True,
            show_display=show_display
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

        offset_x = current_pos['X'] - camera_loc[0] #Quick logic: If camera_loc is (0,0) and current_pos is (-10,-10), then offset_x = -10, we need to
        offset_y = current_pos['Y'] - camera_loc[1]
        target_x = TARGET_PLACE_LOCATION[0] + offset_x #Quick logic: if target_location is (100,100), and offsets are (-10,-10), then target_x = 90
        target_y = TARGET_PLACE_LOCATION[1] + offset_y

        printer.linear_move(x=target_x, y=target_y)
        printer.wait_for_idle()

        printer.linear_move(z=TARGET_PLACE_LOCATION[2])
        printer.wait_for_idle()

        printer.control_solenoid(False)
        printer.control_vacuum(False)
        time.sleep(1.0) #wait for vacuum to release and solenoid to close

        printer.linear_move(z=150)
        printer.wait_for_idle()

        print(f"✓ Target placed at X{target_x:.2f}, Y{target_y:.2f}")

        # Step 4: Find target with Tool 3
        print("\nStep 4: Finding target with Tool 3")
        print("-" * 60)

        printer.select_tool(3)
        printer.wait_for_idle()
        printer.control_led(2, True) #Turn on upper camera LED
        printer.linear_move(x=TARGET_PLACE_LOCATION[0],
                          y=TARGET_PLACE_LOCATION[1],
                          z=TOOL3_CHECK_HEIGHT)
        printer.wait_for_idle()

        # Clear camera buffer after movement and LED turn-on
        time.sleep(0.3)  # Allow LED to stabilize
        for _ in range(5):
            vision_upper.capture_frame()
        time.sleep(0.1)

        def detect_target_upper():
            """Detection callback for target on upper camera."""
            frame = vision_upper.capture_frame()
            if frame is None:
                return None, None, None
            target_pos = vision_upper.find_tool_position()  # Uses target vision params
            if target_pos is not None:
                x, y = target_pos
                return {'X': x, 'Y': y}, None, frame
            return None, None, frame

        success, final_pos_t3 = center_target_in_camera(
            printer=printer,
            vision=vision_upper,
            camera_config=cam2_config,
            detection_method=detect_target_upper,
            tolerance=2,
            max_iterations=20,
            feed_rate=1200,
            debug=True,
            show_display=show_display
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

        # Calculate new offsets using mapped axes (same approach as calibrate_basic_tool)
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
        if not skip_confirm:
            response = input("\nApply new offset to Tool 3? [y/N]: ").strip().lower()
        else:
            response = 'y'

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

        with open("config/tool3_offset.json", 'w') as f:
            json.dump(offset_data, f, indent=2)
        print(f"✓ Tool 3 offset saved to config/tool3_offset.json")

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
