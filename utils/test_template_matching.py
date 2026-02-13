#!/usr/bin/env python3
"""
Template Matching Test Utility for DuetPnP

This script helps diagnose and tune template matching parameters.
It captures frames from a camera and tests template matching in real-time,
showing match scores, detected positions, and rotation angles.

Usage:
    python utils/test_template_matching.py <camera_num> <template_path>

Examples:
    # Test upper camera (2) with component template
    python utils/test_template_matching.py 2 templates/resistor_0402_above.png

    # Test lower camera (0) with component template
    python utils/test_template_matching.py 0 templates/resistor_0402_below.png
"""

import sys
import os
import cv2
import numpy as np
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from machine_vision import VisionTools, CameraConfig, load_camera_config


class TemplateMatchingTester:
    """Interactive template matching tester with live visualization."""

    def __init__(self, camera_number: int, template_path: str):
        """
        Initialize template matching tester.

        Args:
            camera_number: Camera index (0, 2, etc.)
            template_path: Path to template image
        """
        self.camera_number = camera_number
        self.template_path = template_path

        # Load template
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise ValueError(f"Could not load template: {template_path}")

        print(f"Template loaded: {template_path}")
        print(f"Template size: {self.template.shape[1]}x{self.template.shape[0]}")

        # Load camera config
        self.camera_config = load_camera_config(camera_number)

        # Initialize vision tools
        self.vision = VisionTools(camera_number, target='tool', camera_config=self.camera_config)

        # Template matching parameters
        self.match_threshold = 0.6
        self.search_angles = list(range(-15, 16, 1))  # Coarse search angles
        self.exact_angle = None  # If set, only search at this angle

        # Display state
        self.show_template = True
        self.show_rotated = True
        self.freeze_frame = False
        self.frozen_frame = None

    def draw_match_info(self, display: np.ndarray, pos: tuple, angle: int, score: float,
                       template_shape: tuple):
        """Draw match visualization on frame."""
        h, w = template_shape

        if pos is not None:
            # Draw bounding box
            x, y = pos
            cx, cy = x + w//2, y + h//2

            # Draw rectangle around match
            color = (0, 255, 0) if score > self.match_threshold else (0, 165, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)

            # Draw center point
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            # Draw center crosshair
            image_h, image_w = display.shape[:2]
            img_cx, img_cy = image_w // 2, image_h // 2
            cv2.line(display, (img_cx-20, img_cy), (img_cx+20, img_cy), (255, 0, 0), 2)
            cv2.line(display, (img_cx, img_cy-20), (img_cx, img_cy+20), (255, 0, 0), 2)

            # Draw offset line
            cv2.line(display, (cx, cy), (img_cx, img_cy), (255, 255, 0), 2)

            # Calculate offset
            offset_x = cx - img_cx
            offset_y = cy - img_cy
            offset_pixels = np.sqrt(offset_x**2 + offset_y**2)

            # Draw info text
            info_lines = [
                f"Score: {score:.3f} {'MATCH' if score > self.match_threshold else 'NO MATCH'}",
                f"Angle: {angle}°",
                f"Position: ({cx}, {cy})",
                f"Offset: ({offset_x:+d}, {offset_y:+d}) px = {offset_pixels:.1f}px",
                f"Threshold: {self.match_threshold:.2f}"
            ]

            y_pos = 30
            for line in info_lines:
                cv2.putText(display, line, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 30

    def draw_template_window(self, rotated_template: np.ndarray, angle: int, score: float):
        """Show template in separate window."""
        if not self.show_template:
            return

        # Create display for template
        template_display = cv2.cvtColor(rotated_template, cv2.COLOR_GRAY2BGR)

        # Add text
        text = f"Template at {angle}° (score: {score:.3f})"
        cv2.putText(template_display, text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Resize if too small
        h, w = template_display.shape[:2]
        if w < 200 or h < 200:
            scale = max(200/w, 200/h)
            new_w, new_h = int(w*scale), int(h*scale)
            template_display = cv2.resize(template_display, (new_w, new_h),
                                         interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Template (Rotated)', template_display)

    def find_component_with_debug(self, frame: np.ndarray):
        """
        Find component in frame with detailed debug output.

        Returns:
            (position, angle, score, rotated_template)
        """
        if frame is None:
            return None, 0, 0.0, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        best_match = None
        best_score = -1
        best_angle = 0
        best_rotated = None

        # Determine which angles to search
        if self.exact_angle is not None:
            angles_to_search = [self.exact_angle]
        else:
            angles_to_search = self.search_angles

        print(f"\nSearching angles: {angles_to_search}")

        # Search through angles
        for angle in angles_to_search:
            # Rotate template
            h, w = self.template.shape
            matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            matrix[0, 2] += (new_w / 2) - (w / 2)
            matrix[1, 2] += (new_h / 2) - (h / 2)

            rotated = cv2.warpAffine(self.template, matrix, (new_w, new_h))

            # Template matching
            result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            print(f"  Angle {angle:+3d}°: score={max_val:.3f}")

            if max_val > best_score:
                best_score = max_val
                best_match = max_loc
                best_angle = angle
                best_rotated = rotated

        return best_match, best_angle, best_score, best_rotated

    def print_help(self):
        """Print keyboard controls."""
        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        print("="*60)
        print("  SPACE      - Freeze/unfreeze frame")
        print("  t          - Toggle template window")
        print("  r          - Toggle rotated template overlay")
        print("  +/-        - Increase/decrease match threshold")
        print("  [/]        - Decrease/increase angle search range")
        print("  0-9        - Set exact angle (0=0°, 1=45°, 2=90°, etc.)")
        print("  a          - Enable full angular search")
        print("  s          - Save current frame and template")
        print("  h          - Show this help")
        print("  q/ESC      - Quit")
        print("="*60 + "\n")

    def run(self):
        """Run interactive template matching test."""
        self.print_help()

        # Create windows
        cv2.namedWindow('Template Matching Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Template Matching Test', 1280, 960)

        frame_count = 0

        print(f"\nStarting camera {self.camera_number}...")
        print(f"Match threshold: {self.match_threshold:.2f}")

        while True:
            # Capture frame (or use frozen frame)
            if not self.freeze_frame:
                frame = self.vision.capture_frame()
                if frame is None:
                    print("Failed to capture frame")
                    break
                self.frozen_frame = frame.copy()
            else:
                frame = self.frozen_frame.copy()

            # Find component
            pos, angle, score, rotated_template = self.find_component_with_debug(frame)

            # Create display
            display = frame.copy()

            # Draw match info
            if pos is not None and rotated_template is not None:
                self.draw_match_info(display, pos, angle, score, rotated_template.shape)
            else:
                cv2.putText(display, "NO DETECTION", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Add freeze indicator
            if self.freeze_frame:
                cv2.putText(display, "FROZEN", (display.shape[1]-150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Show main window
            cv2.imshow('Template Matching Test', display)

            # Show template window
            if rotated_template is not None:
                self.draw_template_window(rotated_template, angle, score)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord(' '):  # Space - freeze
                self.freeze_frame = not self.freeze_frame
                print(f"Frame {'FROZEN' if self.freeze_frame else 'LIVE'}")
            elif key == ord('t'):  # Toggle template window
                self.show_template = not self.show_template
                if not self.show_template:
                    cv2.destroyWindow('Template (Rotated)')
                print(f"Template window: {'ON' if self.show_template else 'OFF'}")
            elif key == ord('+') or key == ord('='):  # Increase threshold
                self.match_threshold = min(1.0, self.match_threshold + 0.05)
                print(f"Match threshold: {self.match_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):  # Decrease threshold
                self.match_threshold = max(0.0, self.match_threshold - 0.05)
                print(f"Match threshold: {self.match_threshold:.2f}")
            elif key == ord('['):  # Decrease angle range
                step = self.search_angles[1] - self.search_angles[0] if len(self.search_angles) > 1 else 5
                new_step = min(15, step + 1)
                self.search_angles = list(range(-15, 16, new_step))
                print(f"Search angles (step={new_step}): {self.search_angles}")
            elif key == ord(']'):  # Increase angle range
                step = self.search_angles[1] - self.search_angles[0] if len(self.search_angles) > 1 else 5
                new_step = max(1, step - 1)
                self.search_angles = list(range(-15, 16, new_step))
                print(f"Search angles (step={new_step}): {self.search_angles}")
            elif key == ord('a'):  # Full angular search
                self.exact_angle = None
                print("Full angular search enabled")
            elif ord('0') <= key <= ord('9'):  # Set exact angle
                angle_map = {ord('0'): 0, ord('1'): 45, ord('2'): 90, ord('3'): 135,
                            ord('4'): 180, ord('5'): -135, ord('6'): -90, ord('7'): -45}
                if key in angle_map:
                    self.exact_angle = angle_map[key]
                    print(f"Exact angle search: {self.exact_angle}°")
            elif key == ord('s'):  # Save frame
                timestamp = cv2.getTickCount()
                frame_filename = f"debug_frame_{timestamp}.png"
                template_filename = f"debug_template_{timestamp}.png"
                cv2.imwrite(frame_filename, frame)
                cv2.imwrite(template_filename, self.template)
                print(f"Saved: {frame_filename}, {template_filename}")
            elif key == ord('h'):  # Help
                self.print_help()

            frame_count += 1

        cv2.destroyAllWindows()
        print("\nTest completed.")


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print(__doc__)
        print("\nError: Incorrect number of arguments")
        print(f"Usage: {sys.argv[0]} <camera_num> <template_path>")
        sys.exit(1)

    try:
        camera_num = int(sys.argv[1])
        template_path = sys.argv[2]

        if not os.path.exists(template_path):
            print(f"Error: Template file not found: {template_path}")
            sys.exit(1)

        print(f"Starting template matching test...")
        print(f"Camera: {camera_num}")
        print(f"Template: {template_path}")

        tester = TemplateMatchingTester(camera_num, template_path)
        tester.run()

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
