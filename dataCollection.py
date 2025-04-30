import cv2
import numpy as np
import os
from calibration import CalibrateToolheads
from vision_tools import VisionTools
import time
from datetime import datetime

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
        Perform manual calibration to determine mm to pixel ratio and camera bounds.
        User clicks on needle position or presses 'n' if needle is not visible.
        """
        print("\n=== Manual Calibration ===")
        print("1. Move the toolhead to the center of the camera view")
        print("2. Click on the needle position in the image")
        print("3. Press 'n' if the needle is not visible")
        print("4. Press 'q' to finish calibration")
        
        # Move to center position
        self.printer.send_gcode_command("G0 Z80 F6000", check=False)
        self.printer.send_gcode_command(f"G0 X{self.printer.camera_location[0]} Y{self.printer.camera_location[1]} F6000", check=False)
        time.sleep(3.0)
        
        # Initialize calibration data
        calibration_points = []
        pixel_positions = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Get current machine position
                self.printer.send_gcode_command("M114", check=False)
                current_pos = self.printer.parse_position(self.printer.response)
                calibration_points.append((current_pos['X'], current_pos['Y']))
                pixel_positions.append((x, y))
                print(f"Calibration point added: Machine ({current_pos['X']:.2f}, {current_pos['Y']:.2f}) -> Pixel ({x}, {y})")
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        while True:
            frame = self.camera.capture_frame()
            if frame is not None:
                cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                print("Needle not visible - skipping point")
                continue
        
        cv2.destroyAllWindows()
        
        if len(calibration_points) < 2:
            raise ValueError("Need at least 2 calibration points to determine mm to pixel ratio")
        
        # Calculate mm to pixel ratio
        x_ratios = []
        y_ratios = []
        
        for i in range(1, len(calibration_points)):
            dx_mm = calibration_points[i][0] - calibration_points[0][0]
            dy_mm = calibration_points[i][1] - calibration_points[0][1]
            dx_pixel = pixel_positions[i][0] - pixel_positions[0][0]
            dy_pixel = pixel_positions[i][1] - pixel_positions[0][1]
            
            if dx_mm != 0:
                x_ratios.append(abs(dx_pixel / dx_mm))
            if dy_mm != 0:
                y_ratios.append(abs(dy_pixel / dy_mm))
        
        # Use average of ratios
        self.mm_to_pixel = (np.mean(x_ratios) + np.mean(y_ratios)) / 2
        print(f"\nCalibration complete. mm to pixel ratio: {self.mm_to_pixel:.4f}")
        
        # Determine camera bounds
        self.camera_bounds = {
            'min_x': min(p[0] for p in pixel_positions),
            'max_x': max(p[0] for p in pixel_positions),
            'min_y': min(p[1] for p in pixel_positions),
            'max_y': max(p[1] for p in pixel_positions)
        }
        print(f"Camera bounds: {self.camera_bounds}")
    
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