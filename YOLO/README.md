# YOLO Component Detection

YOLO-OBB (Oriented Bounding Box) implementation for fast component detection to replace template matching.

## Quick Start - Weekend Workflow

### Day 1: Data Collection

```bash
cd /Users/sebastian/Code/DuetPnP/YOLO/scripts

# Upper camera (feeder view) - ~300 images
python collect_data_upper.py \
    --template ../../templates/schottky_f_above.png \
    --count 300 \
    --start-x -46.2 \
    --start-y 236.1 \
    --start-z 177

# Lower camera (nozzle view) - ~300 images
python collect_data_lower.py \
    --template ../../templates/schottky_f_below.png \
    --upper-template ../../templates/schottky_f_above.png \
    --count 300 \
    --feeder-x -48.2 \
    --feeder-y 236.1 \
    --feeder-z 15.0 \
    --feeder-focus 17
```

### Day 1 Night: Prepare & Train

```bash
# Prepare dataset (split into train/val)
python prepare_dataset.py --val-split 0.2

# Train on RTX Titan (Linux)
python train.py --epochs 100 --batch 32 --device 0

# Or on P3200 (Windows) - smaller batch size
python train.py --epochs 100 --batch 8 --device 0
```

### Day 2: Export & Test

```bash
# Export to ONNX for Pi (320x320 for speed)
python export_model.py \
    --model ../models/component_obb_*/weights/best.pt \
    --imgsz 320 \
    --simplify

# Benchmark inference time
python test_inference.py --model ../models/best.onnx --benchmark

# Test on validation images
python test_inference.py --model ../models/best.onnx --source ../data/val/images/
```

### Deploy to Pi

```bash
# Copy model to Pi
scp YOLO/models/*/weights/best.onnx pi@<pi-ip>:~/DuetPnP/YOLO/models/

# On Pi - install dependencies
pip install ultralytics onnxruntime
```

---

## Folder Structure

```
YOLO/
├── data/
│   ├── images/          # Training images (auto-generated)
│   └── labels/          # YOLO-OBB labels (auto-generated)
├── models/              # Trained model weights
├── scripts/
│   ├── collect_data_upper.py   # Upper camera data collection
│   └── collect_data_lower.py   # Lower camera data collection (TODO)
├── config/
│   └── dataset.yaml     # Dataset configuration (TODO)
└── README.md
```

## Data Collection

### Upper Camera (Feeder View)

Collects training data by using existing template matching as ground truth:

```bash
cd YOLO/scripts

# Basic usage - will prompt for position
python collect_data_upper.py --template ../../templates/0402_cap_above.png --count 200

# With explicit position
python collect_data_upper.py \
    --template ../../templates/0402_cap_above.png \
    --count 200 \
    --start-x -48.2 \
    --start-y 236.1 \
    --start-z 177

# Customize batch size and jitter
python collect_data_upper.py \
    --template ../../templates/0402_cap_above.png \
    --count 500 \
    --batch-size 50 \
    --jitter 3.0
```

**Options:**
- `--template`: Path to template image (required)
- `--count`: Target number of images (default: 200)
- `--batch-size`: Pause after this many images for manual scene adjustment (default: 20)
- `--jitter`: Random X/Y movement range in mm (default: 2.0)
- `--start-x`, `--start-y`, `--start-z`: Starting camera position
- `--output-dir`: Output directory (default: ../data)

**Controls during collection:**
- `q` - Quit
- `p` - Pause/Resume
- `s` - Skip current frame
- Any key at batch pause - Continue collection

### Lower Camera (Nozzle View)

Collects data with automatic component pickup, C-axis rotation, and X/Y jitter:

```bash
cd YOLO/scripts

# Basic usage with feeder location
python collect_data_lower.py \
    --template ../../templates/0402_cap_below.png \
    --upper-template ../../templates/0402_cap_above.png \
    --count 200 \
    --feeder-x -48.2 \
    --feeder-y 236.1 \
    --feeder-z 20.0 \
    --feeder-focus 177

# Customize images per component and jitter
python collect_data_lower.py \
    --template ../../templates/0402_cap_below.png \
    --upper-template ../../templates/0402_cap_above.png \
    --count 500 \
    --images-per-component 30 \
    --jitter 2.0 \
    --feeder-x -48.2 \
    --feeder-y 236.1 \
    --feeder-z 20.0 \
    --feeder-focus 177
```

**Options:**
- `--template`: Path to lower camera template image (required)
- `--upper-template`: Path to upper camera template for feeder detection (required)
- `--count`: Target number of images (default: 200)
- `--images-per-component`: Captures per picked component (default: 20)
- `--jitter`: Random X/Y movement range in mm (default: 1.5)
- `--feeder-x`, `--feeder-y`, `--feeder-z`: Feeder pickup location (required)
- `--feeder-focus`: Camera focus height above feeder (required)
- `--drop-x`, `--drop-y`: Component drop location (default: feeder location)
- `--output-dir`: Output directory (default: ../data)

**Automatic workflow:**
1. Use upper camera + template matching to find component in feeder
2. Center on component and pick it up
3. Move to lower camera position
4. Capture images with random C-axis rotation (±180°) and X/Y jitter
5. After N images, drop component and pick new one
6. Auto-reset C-axis if cumulative rotation exceeds 360°

**Controls during collection:**
- `q` - Quit
- `p` - Pause/Resume  
- `s` - Skip current frame
- `n` - Drop current component, pick new one

## Label Format

YOLO-OBB uses 4 corner points (normalized 0-1):

```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

Example:
```
0 0.421875 0.385417 0.515625 0.385417 0.515625 0.447917 0.421875 0.447917
```

## Training Pipeline

### 1. Prepare Dataset

After collecting data, split into train/val sets:

```bash
cd YOLO/scripts
python prepare_dataset.py --val-split 0.2
```

This creates:
- `data/train/images/` and `data/train/labels/`
- `data/val/images/` and `data/val/labels/`
- `config/dataset.yaml`

### 2. Train Model

On your GPU machine (RTX Titan or P3200):

```bash
# First install ultralytics
pip install ultralytics

# Train YOLOv8n-OBB (nano - fastest for Pi)
python train.py --epochs 100 --batch 16 --imgsz 640

# For RTX Titan (more VRAM)
python train.py --epochs 100 --batch 32 --imgsz 640

# For P3200 (less VRAM)
python train.py --epochs 100 --batch 8 --imgsz 640
```

Training outputs saved to `YOLO/models/<run_name>/weights/best.pt`

### 3. Export to ONNX

For Raspberry Pi deployment:

```bash
# Standard export (640x640)
python export_model.py --model ../models/<run_name>/weights/best.pt --simplify

# Faster inference (320x320) - recommended for Pi 5
python export_model.py --model ../models/<run_name>/weights/best.pt --imgsz 320 --simplify
```

### 4. Test Inference

```bash
# Test on validation images
python test_inference.py --model ../models/best.onnx --source ../data/val/images/

# Test on live camera
python test_inference.py --model ../models/best.onnx --camera 2

# Benchmark inference time
python test_inference.py --model ../models/best.onnx --benchmark --iterations 100
```

### 5. Deploy to Pi

```bash
# Copy model to Pi
scp YOLO/models/<run_name>/weights/best.onnx pi@<pi-ip>:~/DuetPnP/YOLO/models/

# On Pi, install dependencies
pip install ultralytics onnxruntime
```

## Hardware Requirements

- **Training**: NVIDIA GPU (P3200/RTX Titan recommended)
- **Inference**: Raspberry Pi 5 with ONNX Runtime or ncnn

## Target Performance

- Inference time: <100ms on Raspberry Pi 5
- Model: YOLOv8n-obb (nano)
- Input resolution: 640x640 or 320x320

