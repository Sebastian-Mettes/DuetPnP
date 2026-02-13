#!/usr/bin/env python3
"""
Export trained YOLOv8-OBB model to ONNX for Raspberry Pi deployment.

Prerequisites:
    pip install ultralytics onnx onnxruntime

Usage:
    python export_model.py --model ../models/component_obb_XXXXX/weights/best.pt

For Pi 5 optimization:
    python export_model.py --model best.pt --imgsz 320 --simplify
"""

import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed")
    print("Run: pip install ultralytics")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8-OBB to ONNX')
    parser.add_argument('--model', required=True, help='Path to trained .pt model')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='Image size (320 for faster Pi inference)')
    parser.add_argument('--simplify', action='store_true', 
                        help='Simplify ONNX model (recommended for Pi)')
    parser.add_argument('--half', action='store_true',
                        help='FP16 half precision (not recommended for Pi CPU)')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
    args = parser.parse_args()
    
    # Resolve model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1
    
    print(f"{'='*60}")
    print(f"YOLOv8-OBB Export to ONNX")
    print(f"{'='*60}")
    print(f"Source model: {model_path}")
    print(f"Image size: {args.imgsz}")
    print(f"Simplify: {args.simplify}")
    print(f"Half precision: {args.half}")
    print(f"ONNX opset: {args.opset}")
    print(f"{'='*60}")
    
    # Load model
    print(f"\nLoading model...")
    model = YOLO(str(model_path))
    
    # Export to ONNX
    print(f"\nExporting to ONNX...")
    export_path = model.export(
        format='onnx',
        imgsz=args.imgsz,
        simplify=args.simplify,
        half=args.half,
        opset=args.opset,
    )
    
    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"ONNX model: {export_path}")
    print(f"{'='*60}")
    
    # Print recommended Pi usage
    print(f"""
Raspberry Pi 5 Usage:
---------------------
1. Copy the ONNX model to your Pi:
   scp {export_path} pi@<pi-ip>:~/DuetPnP/YOLO/models/

2. Install onnxruntime on Pi:
   pip install onnxruntime

3. Python inference example:
   from ultralytics import YOLO
   model = YOLO("{Path(export_path).name}")
   results = model.predict(image, imgsz={args.imgsz})

Expected inference time on Pi 5:
  - 640x640: ~150-200ms
  - 320x320: ~50-80ms
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

