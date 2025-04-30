import cv2
import numpy as np
import os
from calibration import CalibrateToolheads
from vision_tools import VisionTools
import time
from datetime import datetime
import json

class DataCollection:
    def __init__(self, toolhead_number: int, output_dir: str = "training_data"):
        """
        Initialize data collection system.
        
        Args:
            toolhead_number (int): Toolhead number to collect data for
            output_dir (str): Directory to save collected images
        """
        self.toolhead_number = toolhead_number
        self.output_dir = output_dir
        self.printer = CalibrateToolheads()
        self.camera = VisionTools()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize calibration parameters
        self.mm_to_pixel = None
        self.camera_bounds = None
        
    def manual_calibration(self):
        """
        Perform manual calibration to determine camera distortion and pixel-to-mm mapping.
        Uses WASD keys for relative movement and stores calibration data.
        """
        print("\n=== Manual Calibration ===")
        print("1. First running automatic camera calibration...")
        
        # Run initial camera calibration
        if not self.printer.calibrate_with_camera(self.toolhead_number):
            raise ValueError("Initial camera calibration failed")
        
        print("\n2. Manual calibration mode:")
        print("- Use WASD keys to move the toolhead (1mm per press)")
        print("- Press SPACE to store current point")
        print("- Press Q to finish after collecting at least 15 points")
        print("\nIMPORTANT: For best results, include points near the edges")
        print("of the visible area. This will maximize the usable camera bounds")
        print("for data collection.")
        
        # Initialize calibration data
        calibration_points = []  # List of (machine_x, machine_y) relative to center
        pixel_points = []  # List of (pixel_x, pixel_y) coordinates
        
        # Store initial center position
        self.printer.send_gcode_command("M114", check=False)
        center_pos = self.printer.parse_position(self.printer.response)
        center_x = center_pos['X']
        center_y = center_pos['Y']
        
        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get current machine position relative to center
                self.printer.send_gcode_command("M114", check=False)
                current_pos = self.printer.parse_position(self.printer.response)
                rel_x = current_pos['X'] - center_x
                rel_y = current_pos['Y'] - center_y
                
                # Store the points
                calibration_points.append((rel_x, rel_y))
                pixel_points.append((x, y))
                print(f"Point stored: Machine({rel_x:.2f}, {rel_y:.2f}) -> Pixel({x}, {y})")
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', on_mouse_click)
        
        while True:
            frame = self.camera.capture_frame()
            if frame is None:
                continue
            
            # Draw stored points on frame
            for px, py in pixel_points:
                cv2.circle(frame, (int(px), int(py)), 3, (0, 255, 0), -1)
            
            cv2.imshow('Calibration', frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses for movement
            if key == ord('w'):  # Forward Y
                self.printer.send_gcode_command("G91", check=False)  # Relative movement
                self.printer.send_gcode_command("G0 Y1 F1200", check=False)
                self.printer.send_gcode_command("G90", check=False)  # Back to absolute
            elif key == ord('s'):  # Backward Y
                self.printer.send_gcode_command("G91", check=False)
                self.printer.send_gcode_command("G0 Y-1 F1200", check=False)
                self.printer.send_gcode_command("G90", check=False)
            elif key == ord('a'):  # Left X
                self.printer.send_gcode_command("G91", check=False)
                self.printer.send_gcode_command("G0 X-1 F1200", check=False)
                self.printer.send_gcode_command("G90", check=False)
            elif key == ord('d'):  # Right X
                self.printer.send_gcode_command("G91", check=False)
                self.printer.send_gcode_command("G0 X1 F1200", check=False)
                self.printer.send_gcode_command("G90", check=False)
            elif key == ord('q'):
                if len(calibration_points) < 15:
                    print(f"Need at least 15 points (currently have {len(calibration_points)})")
                    continue
                break
            
            time.sleep(0.1)  # Small delay to prevent too rapid movement
        
        cv2.destroyAllWindows()
        
        # Calculate camera bounds from pixel points
        self.camera_bounds = {
            'min_x': float(np.min(pixel_points[:, 0])),
            'max_x': float(np.max(pixel_points[:, 0])),
            'min_y': float(np.min(pixel_points[:, 1])),
            'max_y': float(np.max(pixel_points[:, 1]))
        }
        
        # Calculate camera calibration matrix and distortion coefficients
        pixel_points = np.array(pixel_points, dtype=np.float32)
        calibration_points = np.array(calibration_points, dtype=np.float32)
        
        # Reshape points for cv2.findHomography
        src_points = pixel_points.reshape(-1, 1, 2)
        dst_points = calibration_points.reshape(-1, 1, 2)
        
        # Calculate the homography matrix
        H, _ = cv2.findHomography(src_points, dst_points)
        
        # Save calibration data
        calibration_data = {
            'homography_matrix': H.tolist(),
            'center_position': {'x': center_x, 'y': center_y},
            'camera_name': f'camera_{self.toolhead_number}',
            'calibration_date': datetime.now().isoformat(),
            'num_points': len(calibration_points)
        }
        
        # Save to file
        os.makedirs('calibration_data', exist_ok=True)
        calibration_file = os.path.join('calibration_data', f'camera_{self.toolhead_number}_calibration.json')
        with open(calibration_file, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        print(f"\nCalibration complete! Data saved to {calibration_file}")
        print(f"Collected {len(calibration_points)} points")
        
        # Store calibration data in instance
        self.homography_matrix = H
        self.calibration_center = (center_x, center_y)

    def load_camera_calibration(self, camera_name: str) -> bool:
        """
        Load camera calibration data from a saved file.
        
        Args:
            camera_name: Name of the camera calibration to load
            
        Returns:
            bool: True if calibration was loaded successfully
        """
        calibration_file = os.path.join('calibration_data', f'{camera_name}_calibration.json')
        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
                
            self.homography_matrix = np.array(calibration_data['homography_matrix'])
            self.calibration_center = (
                calibration_data['center_position']['x'],
                calibration_data['center_position']['y']
            )
            print(f"Loaded calibration data from {calibration_file}")
            print(f"Calibration date: {calibration_data['calibration_date']}")
            print(f"Number of calibration points: {calibration_data['num_points']}")
            return True
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error loading calibration data: {str(e)}")
            return False
    
    def collect_data(self, num_images: int, grid_size: int = 5):
        """
        Collect training data by moving the toolhead in a grid pattern.
        
        Args:
            num_images (int): Number of images to collect
            grid_size (int): Size of the grid (grid_size x grid_size)
        """
        if self.mm_to_pixel is None or self.camera_bounds is None:
            raise ValueError("Calibration must be performed before data collection")
        
        print("\n=== Data Collection ===")
        print(f"Collecting {num_images} images in a {grid_size}x{grid_size} grid")
        
        # Calculate grid step sizes
        x_step = (self.camera_bounds['max_x'] - self.camera_bounds['min_x']) / (grid_size - 1)
        y_step = (self.camera_bounds['max_y'] - self.camera_bounds['min_y']) / (grid_size - 1)
        
        # Start from center and spiral outward
        center_x = (self.camera_bounds['min_x'] + self.camera_bounds['max_x']) / 2
        center_y = (self.camera_bounds['min_y'] + self.camera_bounds['max_y']) / 2
        
        # Convert pixel positions to machine coordinates
        center_machine_x = self.printer.camera_location[0]
        center_machine_y = self.printer.camera_location[1]
        
        images_collected = 0
        while images_collected < num_images:
            for i in range(grid_size):
                for j in range(grid_size):
                    if images_collected >= num_images:
                        break
                    
                    # Calculate target pixel position
                    target_x = self.camera_bounds['min_x'] + i * x_step
                    target_y = self.camera_bounds['min_y'] + j * y_step
                    
                    # Convert to machine coordinates
                    machine_x = center_machine_x + (target_x - center_x) / self.mm_to_pixel
                    machine_y = center_machine_y + (target_y - center_y) / self.mm_to_pixel
                    
                    # Move to position
                    self.printer.send_gcode_command(f"G0 X{machine_x:.3f} Y{machine_y:.3f} F6000", check=False)
                    time.sleep(1.5)  # Wait for movement to complete
                    
                    # Capture image
                    frame = self.camera.capture_frame()
                    if frame is not None:
                        # Generate filename with position and tool number
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"tool{self.toolhead_number}_x{int(target_x)}_y{int(target_y)}_{timestamp}.jpg"
                        filepath = os.path.join(self.output_dir, filename)
                        
                        # Save image
                        cv2.imwrite(filepath, frame)
                        print(f"Saved image {images_collected + 1}/{num_images}: {filename}")
                        images_collected += 1
        
        print("\nData collection complete!")
    
    def close(self):
        """Clean up resources."""
        self.printer.close()
        self.camera.cleanup()

if __name__ == "__main__":
    # Example usage
    data_collector = DataCollection(toolhead_number=0)
    try:
        data_collector.manual_calibration()
        data_collector.collect_data(num_images=100, grid_size=5)
    finally:
        data_collector.close() 