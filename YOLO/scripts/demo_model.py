#!/usr/bin/env python3
"""
Demo YOLO-OBB model on validation data with ground truth comparison.

Shows each image with:
- Green box: Model prediction
- Blue box: Ground truth label
- Confidence score and detected angle

Usage:
    python demo_model.py --model ../models/component_obb_*/weights/best.pt --data ../data/val/

Controls:
    Space/Enter - Next image
    b - Previous image
    q - Quit
    s - Save current image with annotations
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)


def load_yolo_obb_label(label_path: Path, img_width: int, img_height: int) -> list:
    """
    Load YOLO-OBB label and convert to pixel coordinates.
    
    Returns list of (corners, class_id) tuples where corners is 4 points in pixels.
    """
    labels = []
    
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:  # class_id + 8 coordinates
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:9]]
            
            # Convert normalized coords to pixels
            corners = []
            for i in range(0, 8, 2):
                x = coords[i] * img_width
                y = coords[i+1] * img_height
                corners.append([int(x), int(y)])
            
            labels.append((np.array(corners), class_id))
    
    return labels


def draw_obb(frame: np.ndarray, corners: np.ndarray, color: tuple, 
             thickness: int = 2, label: str = None):
    """Draw oriented bounding box on frame."""
    pts = corners.astype(np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    
    # Draw center point
    center = pts.mean(axis=0).astype(int)
    cv2.circle(frame, tuple(center), 4, color, -1)
    
    # Calculate and display angle
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    if label:
        cv2.putText(frame, label, (center[0] - 30, center[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return angle


def main():
    parser = argparse.ArgumentParser(description='Demo YOLO-OBB model on validation data')
    parser.add_argument('--model', required=True, help='Path to model (.pt or .onnx)')
    parser.add_argument('--data', required=True, help='Path to data directory (with images/ and labels/)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--save-dir', help='Directory to save annotated images')
    args = parser.parse_args()
    
    # Resolve paths
    model_path = Path(args.model)
    data_path = Path(args.data)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1
    
    # Find images and labels directories
    if (data_path / 'images').exists():
        images_dir = data_path / 'images'
        labels_dir = data_path / 'labels'
    else:
        images_dir = data_path
        labels_dir = data_path.parent / 'labels'
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return 1
    
    # Get image files
    image_files = sorted(list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png')))
    
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return 1
    
    print(f"Found {len(image_files)} images")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    model.overrides['conf'] = args.conf
    model.overrides['imgsz'] = args.imgsz
    
    # Create save directory if specified
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create window
    cv2.namedWindow('YOLO-OBB Demo', cv2.WINDOW_NORMAL)
    
    print("\nControls:")
    print("  Space/Enter - Next image")
    print("  b - Previous image")
    print("  q - Quit")
    print("  s - Save current annotated image")
    print("\nLegend:")
    print("  Green box - Model prediction")
    print("  Blue box - Ground truth")
    
    current_idx = 0
    
    while True:
        img_path = image_files[current_idx]
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to load: {img_path}")
            current_idx = (current_idx + 1) % len(image_files)
            continue
        
        img_height, img_width = frame.shape[:2]
        display = frame.copy()
        
        # Run inference
        results = model.predict(frame, verbose=False)
        
        # Draw ground truth (blue)
        gt_labels = load_yolo_obb_label(label_path, img_width, img_height)
        gt_angle = None
        for corners, class_id in gt_labels:
            gt_angle = draw_obb(display, corners, color=(255, 100, 0), thickness=2, 
                               label=f"GT")
        
        # Draw predictions (green)
        pred_angle = None
        pred_conf = None
        num_detections = 0
        
        if results and len(results) > 0 and results[0].obb is not None:
            obb = results[0].obb
            num_detections = len(obb.xyxyxyxy)
            
            for i in range(num_detections):
                corners = obb.xyxyxyxy[i].cpu().numpy()
                conf = float(obb.conf[i])
                pred_conf = conf
                
                pred_angle = draw_obb(display, corners, color=(0, 255, 0), thickness=2,
                                     label=f"{conf:.2f}")
        
        # Add info panel
        info_y = 30
        cv2.putText(display, f"Image {current_idx+1}/{len(image_files)}: {img_path.name}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        cv2.putText(display, f"Detections: {num_detections}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        info_y += 25
        
        if pred_angle is not None and gt_angle is not None:
            angle_diff = abs(pred_angle - gt_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            cv2.putText(display, f"Pred angle: {pred_angle:.1f}° | GT angle: {gt_angle:.1f}° | Diff: {angle_diff:.1f}°", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            info_y += 25
        
        if pred_conf is not None:
            cv2.putText(display, f"Confidence: {pred_conf:.3f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show legend
        legend_y = img_height - 60
        cv2.putText(display, "Green = Prediction", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display, "Blue = Ground Truth", (10, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        
        cv2.imshow('YOLO-OBB Demo', display)
        
        # Handle key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') or key == 13:  # Space or Enter
            current_idx = (current_idx + 1) % len(image_files)
        elif key == ord('b'):
            current_idx = (current_idx - 1) % len(image_files)
        elif key == ord('s'):
            if args.save_dir:
                save_path = save_dir / f"demo_{img_path.name}"
            else:
                save_path = img_path.parent / f"demo_{img_path.name}"
            cv2.imwrite(str(save_path), display)
            print(f"Saved: {save_path}")
    
    cv2.destroyAllWindows()
    print("\nDemo complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

