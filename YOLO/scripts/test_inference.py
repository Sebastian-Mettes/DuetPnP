#!/usr/bin/env python3
"""
Test YOLO-OBB inference on live camera or test images.

Usage:
    # Test on saved images
    python test_inference.py --model ../models/best.onnx --source ../data/val/images/

    # Test on live camera
    python test_inference.py --model ../models/best.onnx --camera 2

    # Benchmark inference time
    python test_inference.py --model ../models/best.onnx --benchmark --iterations 100
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import cv2
import numpy as np
import argparse
import time
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)


def draw_obb_results(frame, results):
    """Draw OBB detection results on frame."""
    display = frame.copy()
    
    if results and len(results) > 0 and results[0].obb is not None:
        obb = results[0].obb
        
        for i in range(len(obb.xyxyxyxy)):
            # Get corner points
            corners = obb.xyxyxyxy[i].cpu().numpy().astype(int)
            
            # Get confidence
            conf = float(obb.conf[i])
            
            # Get class (if multi-class)
            cls = int(obb.cls[i]) if obb.cls is not None else 0
            
            # Draw polygon
            cv2.polylines(display, [corners], isClosed=True, 
                         color=(0, 255, 0), thickness=2)
            
            # Draw center
            center = corners.mean(axis=0).astype(int)
            cv2.circle(display, tuple(center), 5, (0, 0, 255), -1)
            
            # Calculate angle from corners
            # Vector from corner 0 to corner 1
            dx = corners[1][0] - corners[0][0]
            dy = corners[1][1] - corners[0][1]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Add text
            text = f"conf:{conf:.2f} ang:{angle:.0f}"
            cv2.putText(display, text, (center[0] - 40, center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return display


def benchmark_inference(model, frame, iterations=100):
    """Benchmark inference time."""
    print(f"\nBenchmarking inference over {iterations} iterations...")
    
    # Warmup
    for _ in range(10):
        _ = model.predict(frame, verbose=False)
    
    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        _ = model.predict(frame, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{iterations} iterations...")
    
    times = np.array(times)
    print(f"\nBenchmark Results:")
    print(f"  Mean: {times.mean():.1f}ms")
    print(f"  Std:  {times.std():.1f}ms")
    print(f"  Min:  {times.min():.1f}ms")
    print(f"  Max:  {times.max():.1f}ms")
    print(f"  Median: {np.median(times):.1f}ms")
    print(f"  FPS: {1000/times.mean():.1f}")
    
    return times.mean()


def main():
    parser = argparse.ArgumentParser(description='Test YOLO-OBB inference')
    parser.add_argument('--model', required=True, help='Path to model (.pt or .onnx)')
    parser.add_argument('--source', help='Path to image or directory of images')
    parser.add_argument('--camera', type=int, help='Camera number for live inference')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--benchmark', action='store_true', help='Run inference benchmark')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    args = parser.parse_args()
    
    # Validate arguments
    if not args.source and args.camera is None and not args.benchmark:
        print("Error: Provide --source (images) or --camera (live) or --benchmark")
        return 1
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    # Set inference parameters
    model.overrides['conf'] = args.conf
    model.overrides['imgsz'] = args.imgsz
    
    if args.benchmark:
        # Create a test frame for benchmarking
        if args.source:
            source_path = Path(args.source)
            if source_path.is_file():
                frame = cv2.imread(str(source_path))
            else:
                # Get first image from directory
                images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
                if images:
                    frame = cv2.imread(str(images[0]))
                else:
                    print("No images found for benchmark, using synthetic frame")
                    frame = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
        elif args.camera is not None:
            cap = cv2.VideoCapture(args.camera)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("Failed to capture frame for benchmark")
                frame = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
        else:
            frame = np.random.randint(0, 255, (960, 1280, 3), dtype=np.uint8)
        
        benchmark_inference(model, frame, args.iterations)
        return 0
    
    if args.camera is not None:
        # Live camera inference
        print(f"Starting live inference on camera {args.camera}")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Error: Could not open camera {args.camera}")
            return 1
        
        cv2.namedWindow('YOLO-OBB Inference', cv2.WINDOW_NORMAL)
        
        fps_times = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference
                start = time.perf_counter()
                results = model.predict(frame, verbose=False)
                inference_time = (time.perf_counter() - start) * 1000
                fps_times.append(inference_time)
                
                # Keep only last 30 for rolling average
                if len(fps_times) > 30:
                    fps_times.pop(0)
                avg_time = np.mean(fps_times)
                
                # Draw results
                display = draw_obb_results(frame, results)
                
                # Add FPS info
                cv2.putText(display, f"Inference: {inference_time:.0f}ms (avg: {avg_time:.0f}ms)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('YOLO-OBB Inference', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    elif args.source:
        # Image or directory inference
        source_path = Path(args.source)
        
        if source_path.is_file():
            images = [source_path]
        else:
            images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
        
        if not images:
            print(f"No images found in {source_path}")
            return 1
        
        print(f"Processing {len(images)} images...")
        print("Press any key for next image, 'q' to quit")
        
        cv2.namedWindow('YOLO-OBB Inference', cv2.WINDOW_NORMAL)
        
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            # Inference
            start = time.perf_counter()
            results = model.predict(frame, verbose=False)
            inference_time = (time.perf_counter() - start) * 1000
            
            # Draw results
            display = draw_obb_results(frame, results)
            
            # Add info
            cv2.putText(display, f"Inference: {inference_time:.0f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, str(img_path.name), 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('YOLO-OBB Inference', display)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

