#!/usr/bin/env python3
"""
YOLO Training Data Collection - Upper Camera (Feeder View)

Automatically collects and labels training data using existing template matching.
Uses the current vision system as "ground truth" to generate YOLO-OBB labels.

Usage:
    python collect_data_upper.py --template ../templates/0402_cap_above.png --count 200
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
from machine_control import Printer


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


def main():
    parser = argparse.ArgumentParser(description='Collect YOLO-OBB training data from upper camera')
    parser.add_argument('--template', required=True, help='Path to template image')
    parser.add_argument('--count', type=int, default=200, help='Target number of images to collect')
    parser.add_argument('--batch-size', type=int, default=20, help='Pause after this many images')
    parser.add_argument('--jitter', type=float, default=2.0, help='Random X/Y movement range in mm')
    parser.add_argument('--start-x', type=float, help='Starting X position (default: from template config)')
    parser.add_argument('--start-y', type=float, help='Starting Y position (default: from template config)')
    parser.add_argument('--start-z', type=float, help='Focus height (default: from template config)')
    parser.add_argument('--output-dir', default='../data', help='Output directory for images/labels')
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    template_path = Path(args.template)
    if not template_path.is_absolute():
        template_path = script_dir / template_path
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Load template to get dimensions
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Error: Could not load template: {template_path}")
        return 1
    
    template_h, template_w = template.shape[:2]
    print(f"Template loaded: {template_path.name} ({template_w}x{template_h} pixels)")
    
    # Initialize hardware
    print("\nInitializing hardware...")
    printer = Printer(upward_camera_number=0, debug=False)
    
    # Load camera config for upper camera (camera 2)
    camera_config = CameraConfig('config/camera_config_2.json')
    vision = VisionTools(2, target='tool', camera_config=camera_config, debug=False)
    
    # Get image dimensions
    img_width = vision.width
    img_height = vision.height
    print(f"Camera resolution: {img_width}x{img_height}")
    
    # Starting position (can be overridden by args)
    start_x = args.start_x if args.start_x else 0  # Will need to be set
    start_y = args.start_y if args.start_y else 0
    start_z = args.start_z if args.start_z else 177  # Default focus height
    
    if args.start_x is None or args.start_y is None:
        print("\n⚠️  No starting position specified!")
        print("Please provide --start-x and --start-y for feeder location")
        print("Or position the camera manually and press Enter to use current position...")
        input()
        pos = printer.get_current_position()
        start_x = pos['X']
        start_y = pos['Y']
        start_z = pos['Z']
        print(f"Using current position: X{start_x:.2f}, Y{start_y:.2f}, Z{start_z:.2f}")
    
    # Select camera tool and turn on LED
    print("\nSetting up camera...")
    printer.select_tool(3)
    printer.control_led(2, True)
    printer.linear_move(z=start_z, f=3000)
    printer.linear_move(x=start_x, y=start_y, f=6000)
    printer.wait_for_idle()
    time.sleep(0.5)
    
    # Clear camera buffer
    for _ in range(5):
        vision.capture_frame()
    
    # Session ID for unique filenames
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*50}")
    print(f"Data Collection Session: {session_id}")
    print(f"Target: {args.count} images")
    print(f"Batch size: {args.batch_size} (pause for manual adjustment)")
    print(f"X/Y jitter: ±{args.jitter}mm")
    print(f"{'='*50}")
    print("\nControls:")
    print("  'q' - Quit collection")
    print("  'p' - Pause/Resume")
    print("  's' - Skip current frame (don't save)")
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    # Collection loop
    collected = 0
    skipped = 0
    failed = 0
    paused = False
    
    cv2.namedWindow('Data Collection', cv2.WINDOW_NORMAL)
    
    try:
        while collected < args.count:
            # Check for pause after batch
            if collected > 0 and collected % args.batch_size == 0 and not paused:
                print(f"\n{'='*50}")
                print(f"Batch complete! Collected {collected}/{args.count} images")
                print("Please adjust the feeder/scene for variety.")
                print("Press any key to continue (or 'q' to quit)...")
                print(f"{'='*50}")
                
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
            
            # Random jitter movement
            jitter_x = random.uniform(-args.jitter, args.jitter)
            jitter_y = random.uniform(-args.jitter, args.jitter)
            target_x = start_x + jitter_x
            target_y = start_y + jitter_y
            
            printer.linear_move(x=target_x, y=target_y, f=3000)
            printer.wait_for_idle()
            time.sleep(0.2)  # Let vibrations settle
            
            # Clear buffer and capture fresh frame
            for _ in range(3):
                vision.capture_frame()
            
            frame = vision.capture_frame()
            if frame is None:
                print("Failed to capture frame")
                failed += 1
                continue
            
            # Detect component using template matching
            result = vision.find_component(str(template_path), angle=0, exact_angle=False)
            
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
                cv2.putText(display, f"Jitter: ({jitter_x:+.1f}, {jitter_y:+.1f})mm", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show preview
                cv2.imshow('Data Collection', display)
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    print("Skipped frame")
                    skipped += 1
                    continue
                elif key == ord('p'):
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
                    continue
                
                if paused:
                    continue
                
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
                
                if collected % 10 == 0:
                    print(f"Collected {collected}/{args.count} images...")
            
            else:
                # No detection
                cv2.putText(display, "No component detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display, f"Collected: {collected}/{args.count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Data Collection', display)
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                
                failed += 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        
        # Return to start position
        printer.linear_move(x=start_x, y=start_y, f=3000)
        printer.control_led(2, False)
        
        print(f"\n{'='*50}")
        print("Collection Summary:")
        print(f"  Images collected: {collected}")
        print(f"  Frames skipped: {skipped}")
        print(f"  Detection failures: {failed}")
        print(f"  Output directory: {output_dir}")
        print(f"{'='*50}")
        
        vision.cleanup()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

