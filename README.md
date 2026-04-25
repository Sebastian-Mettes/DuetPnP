# DuetPnP

A Python-based pick-and-place (PnP) automation system built on the Duet3D printer platform. This project uses computer vision (OpenCV) for automated component detection, positioning, and placement, enabling precise surface-mount component assembly on conformal surfaces using a modified multi-toolhead printer.
## Features

### Core Functionality
- **Automated calibration** - Vision-based tool offset calibration
- **Component detection** - Template matching and HSV-based detection
- **Multi-camera system** - Dual cameras (upward and downward facing) for precision
- **Pick-and-place workflow** - Complete automation from pickup to placement verification
- **Checkpoint/resume** - Resume operations after interruptions
- **Multi-toolhead support** - Independent calibration for each tool (T0, T1, T2, T3)

### Vision System
- Real-time HSV color thresholding
- Circle detection for tool positioning (Hough Transform)
- Template matching for component identification
- Interactive parameter tuning interface
- Per-camera, per-tool vision parameter storage
- Automatic LED control during calibration

### Hardware Integration
- Duet3D printer control via DSF (Duet Software Framework)
- Dual camera support (Camera 0: upward, Camera 2: downward, Camera 3: tool-mounted)
- Vacuum and solenoid control for component pickup
- Rotary feeder system (3-position, B-axis)
- Automatic axis mapping for complex tool configurations

## Project Structure

```
DuetPnP/
├── config/                          # All configuration files
│   ├── camera_config_*.json         # Camera configurations & transforms
│   ├── machine_config.json          # Machine settings (vacuum, solenoid)
│   ├── placement_config.json        # Component placement configuration
│   ├── vision_params_*.json         # Vision parameters per camera/tool
│   └── tool*_offset.json           # Tool calibration results
│
├── utils/                           # Utility scripts
│   └── generate_tool_vision_params.py  # Vision parameter calibration
│
├── Core Modules:
│   ├── machine_vision.py           # Vision operations (VisionTools, CameraConfig)
│   ├── machine_control.py          # Machine control (Printer, Feeder, centering)
│   └── pnp_operations.py           # PnP workflow (ConfigManager, PnPWorkflow)
│
├── Action Scripts:
│   ├── calibrate_tools.py          # Tool calibration script
│   └── run_pnp_task.py            # Main PnP execution script
│
├── templates/                      # Component template images
├── images/                         # Reference images
└── old_implementation/             # Archived code (DO NOT USE)
```

## Hardware Requirements

- **Computer**: Raspberry Pi or similar (running DSF)
- **Printer**: Duet3D-based pick-and-place machine
  - T0, T1, T2: T2 is a PnP toolheads with vacuum pickup
  - T3: Tool-mounted camera for verification
- **Cameras**:
  - Camera 0: Lower/upward-facing (for viewing tool tips)
  - Camera 2: Upper/downward-facing (for viewing feeders)

- **Feeders**: 3-position rotary feeder (B-axis, 120° rotation)
- **Lighting**: LED ring lights for both cameras
- **Connection**: DSF socket connection to Duet board

## Software Requirements

- **Python**: 3.11 (specified in pyproject.toml)
- **dsf-python**: 3.6rc2 (must match your printer's firmware version)
- **opencv-python**: >=4.11.0.86
- **numpy**: >=2.2.4

> ⚠️ **Important**: The `dsf-python` version must exactly match your RepRapFirmware version. Current version (3.6rc2) works with RepRapFirmware 3.6RC2.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.gatech.edu/smettes3/DuetPnP.git
cd DuetPnP
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Quick Start Guide

### 1. Initial Setup

First, configure camera settings for each camera in `config/camera_config_*.json`:
- Image transformations (rotation, flipping)
- Pixel-to-mm conversion factors
- LED control pins
- Machine location (for upward camera)

### 2. Generate Vision Parameters

Generate vision parameters for each camera/tool combination:

```bash
# For tool detection
python utils/generate_tool_vision_params.py 0 0  # Camera 0 detecting Tool 0
python utils/generate_tool_vision_params.py 0 1  # Camera 0 detecting Tool 1
python utils/generate_tool_vision_params.py 0 2  # Camera 0 detecting Tool 2

# For target detection (used in calibration)
python utils/generate_tool_vision_params.py 0 target  # Camera 0 target
python utils/generate_tool_vision_params.py 2 target  # Camera 2 target
```

**Interactive Calibration:**
1. Script automatically turns on relevant camera LED
2. Three windows appear:
   - **Camera Feed**: Live view with detected circles
   - **HSV Controls**: Adjust color filtering (Hue, Saturation, Value)
   - **Circle Controls**: Adjust detection parameters (dp, minDist, param1, param2, radii)
3. Adjust sliders until target is reliably detected (green circle)
4. Press **'q'** to save parameters
5. LED automatically turns off

Parameters are saved to `config/vision_params_camera{N}_tool{T}.json`

### 3. Calibrate Tools

Calibrate tool offsets using vision-based calibration:

```bash
# Calibrate all tools
python calibrate_tools.py --tools all

# Calibrate specific tools
python calibrate_tools.py --tools 0 1 2

# Calibrate tool 3 (camera tool)
python calibrate_tools.py --tools 3

# Skip confirmations for automation
python calibrate_tools.py --tools all --skip-confirm
```

**What happens during calibration:**
- Tools are centered under the upward camera
- Position offsets are calculated with axis mapping support
- Results saved to `config/tool{N}_offset.json`
- Offsets optionally applied to firmware via G10 commands

### 4. Configure Component Placement

Edit `config/placement_config.json` to define your components:

```json
{
  "components": [
    {
      "type": "resistor_0201",
      "upper_template": "templates/resistor_above.png",
      "lower_template": "templates/resistor_below.png",
      "feed_number": 0,
      "reel_location": {"x": 50.0, "y": 50.0, "z": -8.0},
      "reel_focus": 101.25,
      "feeder_button_location": {"x": 45.0, "y": 55.0, "z": 0.0},
      "placements": [
        {"x": 10.0, "y": 10.0, "z": 0.5, "rotation": 0},
        {"x": 15.0, "y": 10.0, "z": 0.5, "rotation": 90}
      ]
    }
  ]
}
```

**Configuration Fields:**
- `reel_location`: XYZ coordinates where component sits in feeder
- `reel_focus`: Z height for camera focus when viewing component in feeder
- `feeder_button_location`: XYZ coordinates where tool presses to advance feeder (for future automation)
- `placements`: Array of placement locations with rotation

**Note:** Vacuum and solenoid pins are configured in `config/machine_config.json`, not in the placement config.

**Required template images:**
- `upper_template`: Component view from above (in feeder)
- `lower_template`: Component view from below (on tool)

Save template images to the `templates/` directory.

### 5. Run Pick-and-Place

Execute the PnP workflow:

```bash
python run_pnp_task.py config/placement_config.json
```

**Workflow:**
1. Component location (upper camera detects in feeder)
2. Pickup (activate vacuum, lower, grip, lift)
3. Verification (lower camera centers component)
4. Orientation detection (template matching for rotation)
5. Placement (move to target with offset correction)
6. Verification (tool-mounted camera captures placement)

**Resume capability:**
- Progress saved to `config/pnp_checkpoint.json`
- Resume interrupted operations automatically

## Configuration Files

### Camera Configuration

**`config/camera_config_0.json`** (Upward camera):
```json
{
  "name": "Lower Camera (Upward Facing)",
  "machine_location": {"x": 21.6, "y": -59.7, "z": 168.00},
  "transform": {
    "image_flip": "vertical",
    "image_rotate": 180,
    "pixel_to_mm": 0.015,
    "pixel_x_to_machine_x": 0.0,
    "pixel_x_to_machine_y": 1.0,
    "pixel_y_to_machine_x": -1.0,
    "pixel_y_to_machine_y": 0.0
  },
  "led_control": {"pin": "fan4"}
}
```

**Key fields:**
- `machine_location`: Physical location of camera in machine coordinates
- `transform`: Pixel-to-machine coordinate transformation matrix
- `led_control`: LED ring light pin assignment

### Machine Configuration

**`config/machine_config.json`**:
```json
{
  "vacuum": {
    "pin": "fan1",
    "on_value": 40,
    "off_value": 0
  },
  "solenoid": {
    "pin": "fan2",
    "on_value": 255,
    "off_value": 0
  }
}
```

## Advanced Features

### Axis Mapping Support

The system automatically handles tools with different axis mappings (e.g., X→U, Y→V):
1. Queries M563 to determine axis assignments
2. Calculates offsets in source coordinates (X, Y, Z)
3. Applies offsets to correct physical axes
4. Saves axis mapping data in calibration results

### Coordinate Systemssebastian@indirectproof.net

The system maintains several reference frames:
1. **Machine coordinates**: Absolute printer coordinates
2. **Camera reference**: Fixed camera location
3. **Camera offsets**: From calibration
4. **Tool offsets**: G10 offsets per toolhead

All transforms are config-driven (no hardcoded values).

### Template Caching

Vision system caches template images for performance:
- Templates loaded once on initialization
- Grayscale conversion automatic
- Shared across multiple detection calls

## Troubleshooting

### Camera Issues

**Camera not detected:**
- Check camera connection to USB/CSI port
- Verify camera permissions: `ls -l /dev/video*`
- Test camera: `v4l2-ctl --list-devices`

**Poor detection:**
- Ensure adequate lighting (use LED ring lights)
- Check camera focus
- Re-run vision parameter calibration
- Verify HSV thresholds isolate target properly

### Vision System Issues

**Tool not detected:**
1. Check vision parameters: `config/vision_params_camera{N}_tool{T}.json`
2. Re-run parameter generation: `python utils/generate_tool_vision_params.py {N} {T}`
3. Verify LED is turning on during detection
4. Check camera is at correct height/focus

**Template matching fails:**
- Recapture template images with better lighting
- Ensure template shows component clearly
- Adjust grayscale conversion if needed

### Connection Issues

**Duet3D connection fails:**
- Verify DSF is running: `sudo systemctl status duetcontrolserver`
- Check socket connection: `/var/run/dsf/dcs.sock`
- Verify firmware version matches `dsf-python` version
- Review printer logs

### Calibration Issues

**Offsets incorrect:**
- Ensure vision parameters are well-tuned
- Check camera is properly focused
- Verify centering tolerance (default: 2 pixels)
- Review axis mapping in calibration output

**G10 commands not applying:**
- Check tool numbers match firmware configuration
- Verify axis mappings with M563
- Review firmware tool definitions

## Development Notes

### Adding New Components

1. Capture template images (component from above and below)
2. Save to `templates/` directory
3. Add entry to `config/placement_config.json`
4. Define placement locations with coordinates and rotations

### Modifying Detection Parameters

After changing lighting or camera positions:
1. Run `python utils/generate_tool_vision_params.py <camera_num> <tool_num>`
2. Adjust HSV ranges to isolate target
3. Tune circle detection parameters
4. Press 'q' to save

### Debug Mode

Enable debug output in scripts:
```python
printer = Printer(upward_camera_number=0, debug=True)
```

Shows verbose G-code commands and responses.

## Contributing

This is a project for Georgia Tech. Please contact the maintainer for contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Sebastian Mettes**
Georgia Institute of Technology
Email: smettes3@gatech.edu

## Acknowledgments

- Built on Duet3D RepRapFirmware platform
- Uses OpenCV for computer vision
- Integrates dsf-python for machine control