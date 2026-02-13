#!/usr/bin/env python3
"""
Train YOLOv8-OBB model on collected component data.

Prerequisites:
    pip install ultralytics

Usage:
    python train.py --epochs 100 --batch 16

For RTX Titan (24GB VRAM):
    python train.py --epochs 100 --batch 32 --imgsz 640

For P3200 (6GB VRAM):
    python train.py --epochs 100 --batch 8 --imgsz 640
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-OBB model')
    parser.add_argument('--model', default='yolov8n-obb.pt', 
                        help='Base model (default: yolov8n-obb.pt for nano)')
    parser.add_argument('--data', default='../config/dataset.yaml',
                        help='Dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default='0', help='CUDA device (0, 1, etc.) or cpu')
    parser.add_argument('--workers', type=int, default=8, help='Data loader workers')
    parser.add_argument('--patience', type=int, default=20, 
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--name', help='Run name (default: auto-generated)')
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = script_dir / data_path
    
    if not data_path.exists():
        print(f"Error: Dataset config not found: {data_path}")
        print("Run prepare_dataset.py first to create the dataset configuration.")
        return 1
    
    # Generate run name if not provided
    if args.name is None:
        args.name = f"component_obb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Output directory
    output_dir = script_dir.parent / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"YOLOv8-OBB Training")
    print(f"{'='*60}")
    print(f"Base model: {args.model}")
    print(f"Dataset: {data_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Run name: {args.name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"\nLoading base model: {args.model}")
    model = YOLO(args.model)
    
    # Train
    print("\nStarting training...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
        # Augmentation settings optimized for component detection
        hsv_h=0.015,  # Slight hue variation
        hsv_s=0.7,    # Saturation variation
        hsv_v=0.4,    # Value/brightness variation
        degrees=180,  # Full rotation augmentation (components can be any angle)
        translate=0.1,
        scale=0.2,
        flipud=0.5,   # Vertical flip
        fliplr=0.5,   # Horizontal flip
        mosaic=0.5,   # Mosaic augmentation
        mixup=0.1,    # Mixup augmentation
    )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best model saved to: {output_dir}/{args.name}/weights/best.pt")
    print(f"{'='*60}")
    
    # Print final metrics
    if hasattr(results, 'results_dict'):
        print("\nFinal metrics:")
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

