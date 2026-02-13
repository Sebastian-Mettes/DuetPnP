#!/usr/bin/env python3
"""
YOLO Training Data Collection - Lower Camera (Nozzle View)

Automatically collects and labels training data using existing template matching.
Uses the current vision system as "ground truth" to generate YOLO-OBB labels.

This script:
1. Uses upper camera + template matching to find component in feeder
2. Centers on the component and picks it up
3. Moves to lower camera position
4. Captures images at various C-axis rotations and X/Y positions
5. Drops the component and picks a new one
6. Repeats until target count reached

Usage:
    python collect_data_lower.py \
        --template ../../templates/0402_cap_below.png \
        --upper-template ../../templates/0402_cap_above.png \
        --count 200 \
        --feeder-x -48.2 --feeder-y 236.1 --feeder-z 20.0 --feeder-focus 177
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import argparse
import random
import time
from datetime import datetime
from pathlib import Path

from machine_vision import VisionTools, CameraConfig
from machine_control import Printer, center_target_in_camera


def calculate_obb_corners(center_x: float, center_y: float, 
                          width: float, height: float, 
                          angle_deg: float) -> list:
    """
    Calculate the 4 corners of an oriented bounding box.
    
    Args:
        center_x, center_y: Center of the box in pixels
        width, height: Dimensions of the box (before rotation)
        angle_deg: Rotation angle in degrees (counterclockwise)
    
    Returns:
        List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        Starting from top-left, going clockwise
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Half dimensions
    hw = width / 2
    hh = height / 2
    
    # Corner offsets from center (before rotation)
    # Top-left, Top-right, Bottom-right, Bottom-left
    corners_local = [
        (-hw, -hh),
        (hw, -hh),
        (hw, hh),
        (-hw, hh)
    ]
    
    # Rotate and translate to image coordinates
    corners = []
    for dx, dy in corners_local:
        # Rotate
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        # Translate
        corners.append((center_x + rx, center_y + ry))
    
    return corners


def corners_to_yolo_obb(corners: list, img_width: int, img_height: int) -> str:
    """
    Convert corner points to YOLO-OBB format (normalized).
    
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    All coordinates normalized to 0-1
    """
    normalized = []
    for x, y in corners:
        nx = x / img_width
        ny = y / img_height
        # Clamp to valid range
        nx = max(0, min(1, nx))
        ny = max(0, min(1, ny))
        normalized.extend([nx, ny])
    
    # Class 0 (single class for now)
    return f"0 {' '.join(f'{v:.6f}' for v in normalized)}"


def draw_obb(frame: np.ndarray, corners: list, color=(0, 255, 0), thickness=2):
    """Draw oriented bounding box on frame."""
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    # Draw center point
    center_x = int(sum(c[0] for c in corners) / 4)
    center_y = int(sum(c[1] for c in corners) / 4)
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    return frame


def pick_component(printer: Printer, vision_upper: VisionTools, camera_config_upper: CameraConfig,
                   upper_template_path: str, feeder_x: float, feeder_y: float, 
                   feeder_z: float, feeder_focus: float) -> bool:
    """
    Find and pick up a component from the feeder using vision.
    
    Returns:
        True if component was successfully picked, False otherwise
    """
    print("  Finding component in feeder...")
    
    # Switch to camera tool (T3)
    printer.select_tool(3)
    printer.control_led(2, True)  # Upper camera LED
    
    # Move to feeder location at focus height
    printer.linear_move(z=150, f=6000)
    printer.wait_for_idle()
    printer.linear_move(x=feeder_x, y=feeder_y, f=6000)
    printer.linear_move(z=feeder_focus, f=3000)
    printer.wait_for_idle()
    time.sleep(0.3)  # Let camera settle
    
    # Clear camera buffer
    for _ in range(5):
        vision_upper.capture_frame()
    
    # Define detection method for centering
    def detect_component():
        frame = vision_upper.capture_frame()
        if frame is None:
            return None, None, None
        result = vision_upper.find_component(upper_template_path, angle=0, exact_angle=False)
        if result[0] is not None:
            pos, angle = result
            return {'X': pos[0], 'Y': pos[1]}, angle, frame
        return None, None, frame
    
    # Center on component
    success, centered_pos = center_target_in_camera(
        printer=printer,
        vision=vision_upper,
        camera_config=camera_config_upper,
        detection_method=detect_component,
        tolerance=2,
        max_iterations=15,
        feed_rate=1200,
        debug=False,
        show_display=True
    )
    
    if not success:
        print("  Failed to find/center component in feeder")
        printer.control_led(2, False)
        return False
    
    # Save the component location (where camera found it)
    component_x = centered_pos['X']
    component_y = centered_pos['Y']
    print(f"  Component found at X{component_x:.2f}, Y{component_y:.2f}")
    
    # Turn off camera LED
    printer.control_led(2, False)
    
    # Switch to PnP tool (T2) - use fast since we're on T3
    printer.select_tool(2, fast=True)
    
    # Move to safe height first
    printer.linear_move(z=50, f=6000)
    printer.wait_for_idle()
    
    # Move to component location (where camera found it)
    printer.linear_move(x=component_x, y=component_y, f=6000)
    printer.wait_for_idle()
    
    # Turn on vacuum before lowering
    printer.control_vacuum(True)
    
    # Lower to pickup height
    printer.linear_move(z=feeder_z, f=3000)
    printer.wait_for_idle()
    time.sleep(0.2)  # Let vacuum grip
    
    # Lift up
    printer.linear_move(z=50, f=6000)
    printer.wait_for_idle()
    
    print("  Component picked")
    return True


def drop_component(printer: Printer, drop_x: float = None, drop_y: float = None):
    """Drop the current component."""
    print("  Dropping component...")
    
    # Move to drop location if specified
    if drop_x is not None and drop_y is not None:
        printer.linear_move(z=50, f=6000)
        printer.wait_for_idle()
        printer.linear_move(x=drop_x, y=drop_y, f=6000)
        printer.wait_for_idle()
        printer.linear_move(z=20, f=3000)
        printer.wait_for_idle()
    
    # Turn off vacuum and pulse solenoid to release
    printer.control_vacuum(False)
    printer.control_solenoid(True)
    time.sleep(0.1)
    printer.control_solenoid(False)
    
    # Lift up
    printer.linear_move(z=50, f=6000)
    printer.wait_for_idle()
    
    print("  Component dropped")


def main():
    parser = argparse.ArgumentParser(description='Collect YOLO-OBB training data from lower camera')
    parser.add_argument('--template', required=True, help='Path to template image (lower camera view)')
    parser.add_argument('--upper-template', required=True, help='Path to template image (upper camera/feeder view)')
    parser.add_argument('--count', type=int, default=200, help='Target number of images to collect')
    parser.add_argument('--images-per-component', type=int, default=20, 
                        help='Images to capture per picked component')
    parser.add_argument('--jitter', type=float, default=1.5, help='Random X/Y movement range in mm')
    parser.add_argument('--feeder-x', type=float, required=True, help='Feeder X position')
    parser.add_argument('--feeder-y', type=float, required=True, help='Feeder Y position')
    parser.add_argument('--feeder-z', type=float, required=True, help='Feeder pickup Z height')
    parser.add_argument('--feeder-focus', type=float, required=True, help='Camera focus height above feeder')
    parser.add_argument('--drop-x', type=float, help='Drop location X (default: feeder location)')
    parser.add_argument('--drop-y', type=float, help='Drop location Y (default: feeder location)')
    parser.add_argument('--output-dir', default='../data', help='Output directory for images/labels')
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = script_dir / template_path
    
    upper_template_path = Path(args.upper_template)
    if not upper_template_path.is_absolute():
        upper_template_path = script_dir / upper_template_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load lower camera template to get dimensions
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template: {template_path}")
        return 1
    
    template_h, template_w = template.shape[:2]
    print(f"Lower template loaded: {template_path.name} ({template_w}x{template_h} pixels)")
    
    # Load upper camera template (for feeder detection)
    upper_template = cv2.imread(str(upper_template_path), cv2.IMREAD_GRAYSCALE)
    if upper_template is None:
        print(f"Error: Could not load upper template: {upper_template_path}")
        return 1
    print(f"Upper template loaded: {upper_template_path.name}")
    
    # Initialize hardware
    print("\nInitializing hardware...")
    printer = Printer(upward_camera_number=0, debug=False)
    
    # Load camera config for lower camera (camera 0)
    camera_config_lower = CameraConfig('config/camera_config_0.json')
    vision_lower = VisionTools(0, target='tool', camera_config=camera_config_lower, debug=False)
    
    # Load camera config for upper camera (camera 2) - for feeder detection
    camera_config_upper = CameraConfig('config/camera_config_2.json')
    vision_upper = VisionTools(2, target='tool', camera_config=camera_config_upper, debug=False)
    
    # Get image dimensions from lower camera
    img_width = vision_lower.width
    img_height = vision_lower.height
    print(f"Lower camera resolution: {img_width}x{img_height}")
    
    # Get camera location from printer config
    camera_loc = printer.camera_location
    print(f"Lower camera location: X{camera_loc[0]:.2f}, Y{camera_loc[1]:.2f}, Z{camera_loc[2]:.2f}")
    
    # Drop location defaults to feeder location
    drop_x = args.drop_x if args.drop_x else args.feeder_x
    drop_y = args.drop_y if args.drop_y else args.feeder_y
    
    # Initial tool selection will be handled by pick_component
    print("\nReady to start data collection...")
    
    # Session ID for unique filenames
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*50}")
    print(f"Data Collection Session: {session_id}")
    print(f"Target: {args.count} images")
    print(f"Images per component: {args.images_per_component}")
    print(f"X/Y jitter: ±{args.jitter}mm")
    print(f"Feeder location: X{args.feeder_x}, Y{args.feeder_y}, Z{args.feeder_z}")
    print(f"Feeder focus height: {args.feeder_focus}")
    print(f"{'='*50}")
    print("\nControls:")
    print("  'q' - Quit collection")
    print("  'p' - Pause/Resume")
    print("  's' - Skip current frame (don't save)")
    print("  'n' - Drop current component, pick new one")
    print("\nPress Enter to start...")
    input()
    
    # Collection loop
    collected = 0
    skipped = 0
    failed = 0
    components_used = 0
    current_c_angle = 0  # Track cumulative C-axis rotation
    paused = False
    
    cv2.namedWindow('Data Collection', cv2.WINDOW_NORMAL)
    
    try:
        while collected < args.count:
            # Pick up a new component using vision
            components_used += 1
            print(f"\n--- Component #{components_used} ---")
            pick_success = pick_component(
                printer=printer,
                vision_upper=vision_upper,
                camera_config_upper=camera_config_upper,
                upper_template_path=str(upper_template_path),
                feeder_x=args.feeder_x,
                feeder_y=args.feeder_y,
                feeder_z=args.feeder_z,
                feeder_focus=args.feeder_focus
            )
            
            if not pick_success:
                print("  Skipping - failed to pick component")
                continue
            
            # Turn on lower camera LED
            printer.control_led(0, True)
            
            # Move to camera location
            printer.linear_move(x=camera_loc[0], y=camera_loc[1], z=camera_loc[2], f=6000)
            printer.wait_for_idle()
            time.sleep(0.3)  # Let vibrations settle
            
            # Clear camera buffer
            for _ in range(5):
                vision_lower.capture_frame()
            
            # Capture multiple images with this component
            images_this_component = 0
            
            while images_this_component < args.images_per_component and collected < args.count:
                # Random rotation (full 360 degree range for variety)
                rotation_delta = random.uniform(-180, 180)
                printer.rotate_c_axis(rotation_delta, feed_rate=6000)
                current_c_angle += rotation_delta
                
                # Random X/Y jitter
                jitter_x = random.uniform(-args.jitter, args.jitter)
                jitter_y = random.uniform(-args.jitter, args.jitter)
                target_x = camera_loc[0] + jitter_x
                target_y = camera_loc[1] + jitter_y
                
                printer.linear_move(x=target_x, y=target_y, f=2000)
                printer.wait_for_idle()
                time.sleep(0.15)  # Let vibrations settle
                
                # Clear buffer and capture fresh frame
                for _ in range(3):
                    vision_lower.capture_frame()
                
                frame = vision_lower.capture_frame()
                if frame is None:
                    print("Failed to capture frame")
                    failed += 1
                    continue
                
                # Detect component using template matching
                result = vision_lower.find_component(str(template_path), angle=0, exact_angle=False)
                
                display = frame.copy()
                
                if result[0] is not None:
                    center, angle = result
                    center_x, center_y = center
                    
                    # Calculate OBB corners
                    corners = calculate_obb_corners(
                        center_x, center_y,
                        template_w, template_h,
                        angle
                    )
                    
                    # Draw on display
                    draw_obb(display, corners, color=(0, 255, 0), thickness=2)
                    
                    # Add info text
                    cv2.putText(display, f"Detected: angle={angle}deg", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"Collected: {collected}/{args.count}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"Component #{components_used}, img {images_this_component+1}/{args.images_per_component}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, f"C-axis: {current_c_angle:.1f}deg, Jitter: ({jitter_x:+.1f}, {jitter_y:+.1f})mm", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Show preview
                    cv2.imshow('Data Collection', display)
                    key = cv2.waitKey(100) & 0xFF
                    
                    if key == ord('q'):
                        print("Quitting...")
                        raise KeyboardInterrupt
                    elif key == ord('s'):
                        print("Skipped frame")
                        skipped += 1
                        continue
                    elif key == ord('p'):
                        paused = not paused
                        print("PAUSED - press 'p' to resume" if paused else "RESUMED")
                        while paused:
                            key = cv2.waitKey(100) & 0xFF
                            if key == ord('p'):
                                paused = False
                                print("RESUMED")
                            elif key == ord('q'):
                                raise KeyboardInterrupt
                        continue
                    elif key == ord('n'):
                        print("Skipping to next component...")
                        break
                    
                    # Save image and label
                    filename = f"{session_id}_{collected:04d}"
                    img_path = images_dir / f"{filename}.jpg"
                    label_path = labels_dir / f"{filename}.txt"
                    
                    # Save image
                    cv2.imwrite(str(img_path), frame)
                    
                    # Save label in YOLO-OBB format
                    label_str = corners_to_yolo_obb(corners, img_width, img_height)
                    with open(label_path, 'w') as f:
                        f.write(label_str + '\n')
                    
                    collected += 1
                    images_this_component += 1
                    
                    if collected % 10 == 0:
                        print(f"Collected {collected}/{args.count} images...")
                
                else:
                    # No detection
                    cv2.putText(display, "No component detected!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display, "Component may have dropped - press 'n' for new component", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.putText(display, f"Collected: {collected}/{args.count}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Data Collection', display)
                    
                    key = cv2.waitKey(500) & 0xFF
                    if key == ord('q'):
                        raise KeyboardInterrupt
                    elif key == ord('n'):
                        print("User requested new component")
                        break
                    
                    failed += 1
                    
                    # If too many failures, assume component dropped
                    if failed > 5:
                        print("Too many detection failures - picking new component")
                        break
            
            # Drop current component before picking new one
            # Reset C-axis to avoid cable wrap issues
            if abs(current_c_angle) > 360:
                print(f"  Resetting C-axis (was at {current_c_angle:.1f}deg)")
                printer.rotate_c_axis(-current_c_angle, feed_rate=6000)
                current_c_angle = 0
            
            drop_component(printer, drop_x, drop_y)
            failed = 0  # Reset failure counter for next component
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        
        # Clean up - drop any held component
        print("\nCleaning up...")
        printer.control_vacuum(False)
        printer.control_solenoid(True)
        time.sleep(0.1)
        printer.control_solenoid(False)
        
        # Return to safe position
        printer.linear_move(z=50, f=6000)
        printer.control_led(0, False)
        
        print(f"\n{'='*50}")
        print("Collection Summary:")
        print(f"  Images collected: {collected}")
        print(f"  Components used: {components_used}")
        print(f"  Frames skipped: {skipped}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*50}")
        
        vision_lower.cleanup()
        vision_upper.cleanup()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


