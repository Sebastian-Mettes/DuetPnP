#!/usr/bin/env python3
"""
Export trained YOLOv8-OBB model to ONNX for Raspberry Pi deployment.

Prerequisites:
    pip install ultralytics onnx onnxruntime

Usage:
    python export_model.py --model ../models/component_obb_XXXXX/weights/best.pt

For Pi 5 optimization:
    python export_model.py --model best.pt --imgsz 320 --simplify

Verify ONNX export:
    python export_model.py --model best.pt --verify
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

# Try to import ONNX Runtime for verification
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass


def verify_onnx_model(onnx_path: str, imgsz: int = 640):
    """
    Verify ONNX model loads and runs correctly with ONNX Runtime.
    
    Args:
        onnx_path: Path to ONNX model
        imgsz: Image size used during export
    
    Returns:
        True if verification passed, False otherwise
    """
    if not ONNX_AVAILABLE:
        print("⚠️  onnxruntime not installed, skipping verification")
        print("   Install with: pip install onnxruntime")
        return True
    
    import numpy as np
    
    print(f"\nVerifying ONNX model with onnxruntime...")
    
    try:
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
        
        # Get input info
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {input_shape}")
        
        # Create dummy input
        # Shape is typically [batch, channels, height, width]
        if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            h, w = input_shape[2], input_shape[3]
        else:
            h, w = imgsz, imgsz
        
        dummy_input = np.random.randn(1, 3, h, w).astype(np.float32)
        
        # Run inference
        import time
        start = time.perf_counter()
        outputs = session.run(None, {input_name: dummy_input})
        inference_time = (time.perf_counter() - start) * 1000
        
        print(f"  Outputs: {len(outputs)} tensors")
        for i, out in enumerate(outputs):
            print(f"    [{i}] shape: {out.shape}, dtype: {out.dtype}")
        print(f"  Inference time: {inference_time:.1f}ms")
        print("✅ ONNX model verification passed!")
        return True
        
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False


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
    parser.add_argument('--verify', action='store_true',
                        help='Verify exported ONNX model with onnxruntime')
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
    
    # Verify if requested
    if args.verify:
        if not verify_onnx_model(export_path, args.imgsz):
            return 1
    
    # Print recommended Pi usage
    print(f"""
Raspberry Pi 5 Usage:
---------------------
1. Copy the ONNX model to your Pi:
   scp {export_path} pi@<pi-ip>:~/DuetPnP/YOLO/models/

2. Install onnxruntime on Pi:
   pip install onnxruntime

3. Python inference example (using ultralytics):
   from ultralytics import YOLO
   model = YOLO("{Path(export_path).name}")
   results = model.predict(image, imgsz={args.imgsz})

4. Python inference example (using onnxruntime directly - faster):
   from inference_utils import ONNXComponentDetector
   detector = ONNXComponentDetector("{Path(export_path).name}", imgsz={args.imgsz})
   result = detector.detect(frame, expected_angle=0)

Expected inference time on Pi 5:
  - 640x640: ~150-200ms (ultralytics), ~100-150ms (onnxruntime)
  - 320x320: ~50-80ms (ultralytics), ~30-50ms (onnxruntime)
""")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


