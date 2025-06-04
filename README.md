# DuetPnP

A Python library for controlling a pick and place machine built on a Duet3D printer platform. This project integrates computer vision capabilities using OpenCV for automated tool calibration and component placement. It enables precise component placement through camera-based calibration and real-time vision processing.

## Features

### Core Functionality
- Computer vision-based tool calibration
- HSV-based component detection
- Integration with Duet3D control system
- Real-time camera feed processing
- Tool offset calculation and management
- Support for multiple toolheads

### Vision System
- Real-time HSV color thresholding
- Circle detection for tool positioning
- Interactive parameter tuning interface
- Tool position tracking and validation
- Camera feed visualization

### Hardware Integration
- Duet3D printer control
- Camera module support
- Multi-toolhead management
- Precise motion control
- Automatic tool offset compensation

## Hardware Requirements

- Raspberry Pi (recommended)
- Duet3D printer with pick and place modifications
- Camera module (compatible with Raspberry Pi)
- USB connection between Pi and Duet3D board
- Adequate lighting for vision system

## Software Requirements

- Python 3.x
- OpenCV
- NumPy
- dsf-python (version matching your printer's firmware)

## Installation

1. Clone the repository:
```bash
git clone https://github.gatech.edu/smettes3/DuetPnP.git
cd DuetPnP
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Important Note**: The version of `dsf-python` in requirements.txt may need to be modified to match your printer's firmware version. The current version (3.6rc2) works with RepRapFirmware 3.6RC2. If you have a different firmware version, please adjust the dsf-python version accordingly in `requirements.txt`.

## Hardware Setup

1. Connect your camera module to the Raspberry Pi
2. Ensure proper lighting for the vision system
3. Connect the Duet3D board to the Raspberry Pi via USB
4. Verify all toolheads are properly mounted and functional
5. Check camera focus and positioning

## Usage

### Calibration

The `calibration.py` module provides tools for calibrating tool offsets:

```python
from calibration import CalibrateToolheads

calibrator = CalibrateToolheads()
calibrator.home()  # Home all axes first
calibrator.calibrate_with_camera(0)  # Calibrate tool 0
calibrator.calibrate_with_camera(1)  # Calibrate tool 1
calibrator.close()  # Clean up when done
```

### Vision Tools Setup

The `vision_tools.py` module contains OpenCV pipelines for tool detection and calibration:

1. Run the vision tools setup script:
```bash
python vision_tools.py
```

2. This will open three windows:
   - Camera Feed: Shows the live camera view with detected circles
   - HSV Controls: Sliders to adjust the color filtering
   - Circle Controls: Parameters for circle detection

3. Adjust the HSV thresholds to isolate your tool:
   - H (Hue): Color
   - S (Saturation): Color intensity
   - V (Value): Brightness

4. Fine-tune circle detection parameters:
   - dp: Accumulator resolution
   - minDist: Minimum distance between circles
   - param1: Edge detection threshold
   - param2: Circle detection threshold
   - minRadius/maxRadius: Size constraints

5. Press 'q' to save your parameters when satisfied.

## Configuration

- Camera parameters can be adjusted in real-time
- HSV thresholds can be tuned for specific components
- Tool offsets are automatically calculated and stored
- Configuration files are saved in the project directory

## Troubleshooting

### Common Issues

1. Camera not detected:
   - Check camera connection
   - Verify camera permissions
   - Test camera with `raspistill` command

2. Vision system issues:
   - Ensure adequate lighting
   - Check camera focus
   - Verify HSV thresholds
   - Adjust circle detection parameters

3. Duet3D connection issues:
   - Verify USB connection
   - Check firmware version compatibility
   - Ensure proper permissions

## Contributing

This is a project for Georgia Tech. Please contact the maintainer for contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Sebastian Mettes  
Georgia Institute of Technology  
Email: smettes3@gatech.edu
