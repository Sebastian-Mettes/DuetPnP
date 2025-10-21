# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DuetPnP is a Python-based pick-and-place (PnP) automation system built on the Duet3D printer platform. It uses computer vision (OpenCV) for automated component detection, positioning, and placement. The system operates a multi-toolhead printer modified for surface-mount component assembly, using dual cameras (upper downward-facing, lower upward-facing) for precise positioning and verification.

## ⚠️ IMPORTANT: Refactored Architecture (October 2024)

The codebase has been **completely refactored** into a clean, modular structure. Old implementation files have been moved to `old_implementation/` folder for reference.

### New File Structure (USE THESE):
```
├── machine_vision.py          # All vision operations (VisionTools, CameraConfig)
├── machine_control.py         # All machine control (Printer, Feeder, centering)
├── pnp_operations.py          # PnP workflow (ConfigManager, PnPWorkflow)
├── run_pnp_task.py           # Main entry point
├── calibrate_tools.py        # Tool calibration script
├── config/                   # All configuration files
│   ├── camera_config_0.json      # Camera 0 configuration & transforms
│   ├── camera_config_2.json      # Camera 2 configuration & transforms
│   ├── machine_config.json       # Machine settings (vacuum, solenoid pins)
│   ├── placement_config.json     # Component placement configuration
│   ├── vision_params_*.json      # Vision parameters for each camera/tool
│   ├── tool*_offset.json         # Tool calibration results
│   └── camera_offset.json        # Camera offset calibration
├── utils/                    # Utility/helper scripts
│   └── generate_tool_vision_params.py  # Vision parameter calibration tool
├── templates/                # Component template images
├── images/                   # Reference images
└── old_implementation/       # ⚠️ OLD CODE - DO NOT MODIFY
    ├── vision_tools.py       # (archived)
    ├── calibration.py        # (archived)
    ├── feed.py              # (archived)
    ├── PnP.py               # (archived)
    └── ...
```

### Key Architectural Improvements:

1. **Camera Configuration System**: Camera-specific coordinate transforms in JSON files
   - Eliminates hardcoded pixel→mm conversions
   - Automatic image rotation/flipping for logical orientation
   - Per-camera LED control pins

2. **Unified Centering Algorithm**: Single `center_target_in_camera()` function
   - Replaces duplicate centering code from 4 old files
   - Config-driven coordinate transforms
   - Works for tool calibration AND component detection

3. **Tool-Specific Vision Parameters**: Per-(camera, tool) calibration
   - Format: `config/vision_params_camera{N}_tool{T}.json`
   - Eliminates parameter sharing across tools
   - Use `utils/generate_tool_vision_params.py` to create

4. **Clean Separation of Concerns**:
   - `machine_vision.py` → "How to see things"
   - `machine_control.py` → "How to move and center things"
   - `pnp_operations.py` → "What to do with PnP workflows"
   - `run_pnp_task.py` → "Execute the task"

### Running PnP Tasks:
```bash
# Generate tool vision parameters (do once per tool):
python utils/generate_tool_vision_params.py 0 0  # Camera 0, Tool 0
python utils/generate_tool_vision_params.py 0 1  # Camera 0, Tool 1
python utils/generate_tool_vision_params.py 0 2  # Camera 0, Tool 2

# Calibrate tools:
python calibrate_tools.py --tools all        # Calibrate all tools
python calibrate_tools.py --tools 0 1 2      # Calibrate specific tools

# Run pick-and-place:
python run_pnp_task.py config/placement_config.json
```

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Python Version
- **Required**: Python 3.11 (specified in pyproject.toml as `>=3.11,<3.12`)
- The project uses a virtual environment at `.venv/`

### Key Dependencies
- `dsf-python==3.6rc2` - Duet Software Framework interface (must match printer firmware version)
- `opencv-python>=4.11.0.86` - Computer vision processing
- `numpy>=2.2.4` - Array operations for image processing

**Important**: The `dsf-python` version must match your printer's RepRapFirmware version. The current version (3.6rc2) works with RepRapFirmware 3.6RC2.

## Core Architecture (New Refactored System)

### System Components

The refactored system consists of four main modules:

1. **machine_vision.py** - All computer vision operations
   - `VisionTools` class: Frame capture, template matching, circle detection
   - `CameraConfig` class: Camera-specific transforms and configuration
   - Window management functions
   - Template caching for performance
   - Tool-specific vision parameter loading

2. **machine_control.py** - All machine control operations
   - `Printer` class: G-code commands, motion control, tool management
   - `Feeder` class: Rotary belt component feeding
   - `center_target_in_camera()`: Generic centering algorithm (replaces all duplicate code)

3. **pnp_operations.py** - PnP workflow orchestration
   - `ConfigManager`: Loads placement configs, templates, camera offsets
   - `PnPWorkflow`: Orchestrates pickup, orientation detection, and placement

4. **run_pnp_task.py** - Main entry point
   - Initializes all systems
   - Loads camera configurations
   - Executes PnP workflow

### Old Architecture (Archived)

The original system (now in `old_implementation/`) had these files:
- `PnP.py`, `calibration.py`, `vision_tools.py`, `feed.py` - ⚠️ DO NOT USE
- These contained duplicate code and hardcoded transforms
- Kept for reference only

### Hardware Configuration

**Cameras:**
- Camera 0: Lower/upward-facing camera (for viewing tool tips and held components)
- Camera 2: Upper/downward-facing camera (for viewing components in feeders and placement verification)
- Camera 3: Tool-mounted camera (controlled as toolhead T3)

**Toolheads:**
- T0, T1, T2: Pick-and-place tools with vacuum and solenoid control
- T3: Upper camera tool for placement verification

**Control Pins:**
- Fan 1 (`fan1`): Vacuum pump control
- Fan 2 (`fan2`): Solenoid valve control
- Fan 3 (`fan3`): Upper camera ring light
- Fan 4 (`fan4`): Lower camera ring light

**Feeder System:**
- 3-position rotary feeder (B-axis)
- 120° rotation between positions
- Radius: 11.45mm
- Rock-back-then-forward feed mechanism

### Camera Configuration System (NEW)

The refactored system uses JSON configuration files for each camera, eliminating hardcoded coordinate transforms:

**config/camera_config_0.json** (Upward camera):
```json
{
  "transform": {
    "image_flip": "vertical",
    "image_rotate": 180,
    "pixel_to_mm": 0.015,
    "pixel_x_to_machine_x": 0.0,
    "pixel_x_to_machine_y": 1.0,
    "pixel_y_to_machine_x": -1.0,
    "pixel_y_to_machine_y": 0.0
  }
}
```

**config/camera_config_2.json** (Downward camera):
```json
{
  "transform": {
    "image_flip": null,
    "image_rotate": 0,
    "pixel_to_mm": 0.015,
    "pixel_x_to_machine_x": -1.0,
    "pixel_x_to_machine_y": 0.0,
    "pixel_y_to_machine_x": 0.0,
    "pixel_y_to_machine_y": 1.0
  }
}
```

**Benefits:**
- Images automatically rotated/flipped for logical orientation
- All cameras use same convention: object offset +X in image → move -X on machine
- Transform matrix handles camera orientation differences
- Easy to add new cameras - just create new config file

### Coordinate Systems and Offsets

The system maintains several coordinate reference frames:

1. **Machine coordinates** - Absolute printer coordinates (X, Y, Z)
2. **Camera reference point** - Fixed location at `[21.6, -59.7, 166.41]` (stored in `calibration.py`)
3. **Camera offsets** - Stored in `camera_offset.json` from calibration routine
4. **Tool offsets** - G10 offsets for each toolhead, automatically calculated during calibration
5. **Fixed camera offsets** - Manually determined values:
   - Lower camera: `(0, 0)`
   - Upper camera: `(0.85, 0.40)`

### Vision System Architecture

**Detection Strategy:**
The system uses two complementary vision approaches:
- **Circle detection** (Hough Transform): For tools and camera lenses during calibration
- **Template matching**: For component identification and orientation detection

**Parameter Files (NEW - Tool-Specific):**
Vision parameters are now stored per (camera, tool) pair:
- Format: `config/vision_params_camera{N}_tool{T}.json`
- Examples:
  - `config/vision_params_camera0_tool0.json` - Camera 0 detecting Tool 0
  - `config/vision_params_camera0_tool1.json` - Camera 0 detecting Tool 1
  - `config/vision_params_camera0_tool2.json` - Camera 0 detecting Tool 2
  - `config/vision_params_camera2_target.json` - Camera 2 target detection
- Each contains HSV thresholds and circle detection parameters
- Generate using: `python utils/generate_tool_vision_params.py <camera_num> <tool_num>`

**Old Format (Archived):**
- `vision_params_camera0_tool.json` - ⚠️ Shared across all tools (problematic)
- Now in `old_implementation/` folder

**HSV Thresholding:**
The system isolates targets using HSV color space filtering before detection to handle varying lighting conditions.

### Pick-and-Place Workflow

The main workflow in `PnP.py` follows this sequence:

1. **Component Location Phase** (Upper Camera):
   - Move camera to reel location from config
   - Detect component using upper template
   - Iteratively center component (tolerance: 2 pixels)
   - Calculate pickup coordinates with camera offset compensation

2. **Pickup Phase**:
   - Switch to PnP tool (T2)
   - Move to pickup location
   - Activate vacuum pump
   - Lower to pickup height
   - Open solenoid valve
   - Lift to safe height

3. **Verification and Orientation Phase** (Lower Camera):
   - Move to lower camera location
   - Center component in lower camera view
   - Determine component rotation using template matching
   - Rotate component to desired orientation
   - Calculate offset from camera center for placement correction

4. **Placement Phase**:
   - Move to placement location (with offset correction)
   - Lower to placement height
   - Deactivate solenoid valve (release component)
   - Lift to safe height
   - Deactivate vacuum pump

5. **Verification Phase**:
   - Switch to camera tool (T3)
   - Move to placement location
   - Capture verification image

### Configuration Files

**config/placement_config.json** - Main component placement configuration:
```json
{
  "components": [
    {
      "type": "component_name",
      "upper_template": "path/to/above_view.png",
      "lower_template": "path/to/below_view.png",
      "feed_number": 0,
      "reel_location": {"x": 0, "y": 0, "z": 0},
      "reel_focus": 101.25,
      "placements": [
        {"x": 0, "y": 0, "z": 0, "rotation": 0}
      ]
    }
  ],
  "vacuum_pin": "fan1",
  "solenoid_pin": "fan2"
}
```

**Key Fields:**
- `upper_template`: Image of component as seen from above (for feeder detection)
- `lower_template`: Image of component as seen from below (for orientation detection)
- `reel_focus`: Z-height for camera focus at feeder location
- `rotation`: Desired component rotation in degrees

## Common Development Commands

### Running the System

```bash
# Full pick-and-place operation
python PnP.py
```

### Calibration Procedures

```bash
# Calibrate all toolheads and cameras
python calibration.py
```

The calibration script runs this sequence:
1. Calibrate tools 0, 1, 2 using lower camera (camera 0)
2. Calibrate upper camera tool (tool 3) using both cameras
3. Run PnP-based camera calibration for precision

### Vision Parameter Tuning

```bash
# Configure vision parameters for camera 0 detecting tools
python vision_tools.py 0 tool

# Configure vision parameters for camera 2 detecting tools
python vision_tools.py 2 tool

# Configure for camera detecting the camera lens itself
python vision_tools.py 0 camera
```

This opens interactive windows:
- **HSV Controls**: Adjust color filtering (Hue, Saturation, Value ranges)
- **Circle Controls**: Adjust circle detection parameters (dp, minDist, param1, param2, radii)
- **Camera Feed**: Live preview with detected circles

Press 'q' to save parameters and exit.

### Feeder Control

```bash
# Home the feeder system
python feed.py home

# Feed from specific belt position (0-2)
python feed.py 0
```

## Important Implementation Details

### G-code Validation
`calibration.py` includes a whitelist of valid G-code commands in the `VALID_GCODES` list. Any command sent through `send_gcode_command(check=True)` is validated against this list.

### Motion Safety
- The `move_camera()` method in `PnP.py` includes safety checks to prevent camera crashes
- Camera must be above Z=100 before XY movements
- Tool changes have built-in delays (3.5s for T3 camera tool)

### Pixel-to-MM Conversion
The system uses an empirically determined conversion factor of `0.015` pixels/mm for centering operations. This is defined in `PnP.py` as `INITIAL_PIXELS_TO_MM`.

### Component Detection Optimization
`find_component()` in `vision_tools.py` uses a coarse-to-fine search strategy:
1. Coarse search: 5-degree angle steps on 50% downsampled image
2. Fine search: 2-degree steps on full resolution around best coarse match
3. Early termination if match score exceeds 0.95

### Template Matching Threshold
Components are considered detected when template match score exceeds 0.6 in `find_component()`.

### Centering Tolerance
The system considers a component "centered" when within 2 pixels of the image center in both X and Y directions.

### Movement Synchronization
`send_gcode_and_wait()` in `PnP.py` waits for movement completion by:
- Polling M114 position
- Checking for 3 consecutive stable position readings
- Position change tolerance: 0.001mm

## Axis Mapping System

The system handles complex axis mappings where toolheads may use different axis labels (X/Y/Z vs U/V/Z). The `parse_axis_mapping()` and `set_tool_offset()` methods in `calibration.py` handle this automatically by:
1. Querying M563 for tool axis mappings
2. Calculating offsets in the mapped coordinate frame
3. Applying offsets to the correct physical axes

## Development Notes

### Adding New Components
1. Capture template images: component from above (in feeder) and below (on tool)
2. Save template images to `templates/` directory
3. Add component entry to `config/placement_config.json`
4. Tune vision parameters using `utils/generate_tool_vision_params.py` if needed
5. Define placement locations with coordinates and rotations

### Modifying Detection Parameters
Vision parameters are camera and target specific. After changing lighting or camera positions:
1. Run `python utils/generate_tool_vision_params.py <camera_num> <tool_num>`
2. Adjust HSV ranges to isolate the target
3. Tune circle detection parameters until consistent detection
4. Press 'q' to save - parameters stored in `config/vision_params_camera{N}_tool{T}.json`

### Coordinate Debugging
Use `M114` commands to check current position. The `parse_position()` method extracts X, Y, Z coordinates from the response string format: `X:123.45 Y:67.89 Z:10.11`

### Camera Buffer Management
`capture_frame()` supports a `clear_buffer` parameter, but this should be used sparingly. The system is optimized to work without continuous buffer clearing for performance.

## Known Constraints

- Python version locked to 3.11.x (not compatible with 3.12+)
- dsf-python version must exactly match printer firmware
- Camera resolution preference: 1280x960, falls back to 640x480 or 1920x1080
- Maximum centering iterations: 20 attempts
- Template matching requires grayscale images
- System assumes calibration reference point at `[21.6, -59.7, 166.41]`
