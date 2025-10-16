#!/usr/bin/env python3
"""
Tool Vision Parameters Generator

This script helps generate tool-specific vision parameters for each (camera, tool) pair.
It runs an interactive calibration to tune HSV thresholds and circle detection parameters.

Usage:
    python generate_tool_vision_params.py <camera_number> <tool_number>

Examples:
    python generate_tool_vision_params.py 0 0  # Calibrate camera 0 for tool 0
    python generate_tool_vision_params.py 0 1  # Calibrate camera 0 for tool 1
    python generate_tool_vision_params.py 0 2  # Calibrate camera 0 for tool 2

Output:
    Creates vision_params_camera{N}_tool{T}.json with calibrated parameters
"""

import cv2
import numpy as np
import json
import sys
from machine_vision import VisionTools, load_camera_config


def calibrate_tool_vision(camera_number: int, tool_number: int):
    """
    Interactive calibration for tool-specific vision parameters.

    Args:
        camera_number: Camera ID (0, 2, etc.)
        tool_number: Tool number (0, 1, 2, etc.)
    """
    output_file = f"vision_params_camera{camera_number}_tool{tool_number}.json"

    print("="*60)
    print(f"Tool Vision Parameters Calibration")
    print(f"Camera {camera_number}, Tool {tool_number}")
    print("="*60)

    # Load existing params if available
    try:
        with open(output_file, 'r') as f:
            params = json.load(f)
        print(f"Loaded existing parameters from {output_file}")
    except FileNotFoundError:
        # Default parameters
        params = {
            'hsv_lower': [0, 0, 0],
            'hsv_upper': [180, 255, 255],
            'circle_params': {
                'dp': 1,
                'minDist': 20,
                'param1': 100,
                'param2': 20,
                'minRadius': 10,
                'maxRadius': 20
            }
        }
        print("Using default parameters")

    # Load camera config for image transforms
    try:
        camera_config = load_camera_config(camera_number)
        print(f"Loaded camera configuration: {camera_config.config['name']}")
    except:
        print(f"Warning: Could not load camera config for camera {camera_number}")
        camera_config = None

    # Initialize vision with camera config
    vision = VisionTools(camera_number, target='tool', tool_number=tool_number,
                        camera_config=camera_config)

    # Create windows
    cv2.namedWindow('HSV Controls')
    cv2.namedWindow('Circle Controls')
    cv2.namedWindow('Camera Feed')

    # Create HSV trackbars
    cv2.createTrackbar('H min', 'HSV Controls', params['hsv_lower'][0], 180, lambda x: None)
    cv2.createTrackbar('S min', 'HSV Controls', params['hsv_lower'][1], 255, lambda x: None)
    cv2.createTrackbar('V min', 'HSV Controls', params['hsv_lower'][2], 255, lambda x: None)
    cv2.createTrackbar('H max', 'HSV Controls', params['hsv_upper'][0], 180, lambda x: None)
    cv2.createTrackbar('S max', 'HSV Controls', params['hsv_upper'][1], 255, lambda x: None)
    cv2.createTrackbar('V max', 'HSV Controls', params['hsv_upper'][2], 255, lambda x: None)

    # Create circle detection trackbars
    cv2.createTrackbar('dp', 'Circle Controls', int(params['circle_params']['dp']*10), 10, lambda x: None)
    cv2.createTrackbar('minDist', 'Circle Controls', params['circle_params']['minDist'], 100, lambda x: None)
    cv2.createTrackbar('param1', 'Circle Controls', params['circle_params']['param1'], 200, lambda x: None)
    cv2.createTrackbar('param2', 'Circle Controls', params['circle_params']['param2'], 100, lambda x: None)
    cv2.createTrackbar('minRadius', 'Circle Controls', params['circle_params']['minRadius'], 399, lambda x: None)
    cv2.createTrackbar('maxRadius', 'Circle Controls', params['circle_params']['maxRadius'], 400, lambda x: None)

    print("\nInstructions:")
    print("1. Adjust HSV sliders to isolate the tool in the HSV Controls window")
    print("2. Adjust Circle Controls to detect the tool circle")
    print("3. Press 'q' to save parameters and exit")
    print("4. Green circles indicate detected tools")
    print()

    try:
        while True:
            # Get current trackbar values
            hsv_lower = np.array([
                cv2.getTrackbarPos('H min', 'HSV Controls'),
                cv2.getTrackbarPos('S min', 'HSV Controls'),
                cv2.getTrackbarPos('V min', 'HSV Controls')
            ])

            hsv_upper = np.array([
                cv2.getTrackbarPos('H max', 'HSV Controls'),
                cv2.getTrackbarPos('S max', 'HSV Controls'),
                cv2.getTrackbarPos('V max', 'HSV Controls')
            ])

            circle_params = {
                'dp': cv2.getTrackbarPos('dp', 'Circle Controls') / 10.0,
                'minDist': cv2.getTrackbarPos('minDist', 'Circle Controls'),
                'param1': cv2.getTrackbarPos('param1', 'Circle Controls'),
                'param2': cv2.getTrackbarPos('param2', 'Circle Controls'),
                'minRadius': cv2.getTrackbarPos('minRadius', 'Circle Controls'),
                'maxRadius': cv2.getTrackbarPos('maxRadius', 'Circle Controls')
            }

            # Capture and process frame
            frame = vision.capture_frame()
            if frame is None:
                continue

            # Convert to HSV and create mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

            # Apply mask and convert to gray
            masked = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

            # Detect circles
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=circle_params['dp'],
                minDist=circle_params['minDist'],
                param1=circle_params['param1'],
                param2=circle_params['param2'],
                minRadius=circle_params['minRadius'],
                maxRadius=circle_params['maxRadius']
            )

            # Draw circles on frame
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    # Draw circle
                    cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # Draw center
                    cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)
                    # Add radius text
                    cv2.putText(gray, f"r={i[2]}", (i[0]+10, i[1]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display
            cv2.imshow('Camera Feed', gray)

            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # Save parameters
                params = {
                    'hsv_lower': hsv_lower.tolist(),
                    'hsv_upper': hsv_upper.tolist(),
                    'circle_params': circle_params
                }

                with open(output_file, 'w') as f:
                    json.dump(params, f, indent=4)

                print(f"\n✓ Parameters saved to {output_file}")
                print(f"  HSV Lower: {hsv_lower}")
                print(f"  HSV Upper: {hsv_upper}")
                print(f"  Circle params: {circle_params}")
                break

    finally:
        cv2.destroyAllWindows()
        vision.cleanup()


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print(__doc__)
        return 1

    try:
        camera_number = int(sys.argv[1])
        tool_number = int(sys.argv[2])
    except ValueError:
        print("Error: camera_number and tool_number must be integers")
        print(__doc__)
        return 1

    calibrate_tool_vision(camera_number, tool_number)
    return 0


if __name__ == "__main__":
    sys.exit(main())
