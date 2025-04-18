# DuetPnP

A Python library for controlling a pick and place machine built on a Duet3D printer platform. This project integrates computer vision capabilities using OpenCV for automated tool calibration and component placement.

## Features

- Computer vision-based tool calibration
- HSV-based component detection
- Integration with Duet3D control system
- Real-time camera feed processing
- Tool offset calculation and management
- Support for multiple toolheads

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Raspberry Pi (recommended)
- Duet3D printer with pick and place modifications
- Camera module

## Installation

1. Clone the repository:
```bash
git clone https://github.gatech.edu/smettes3/DuetPnP.git
cd DuetPnP
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Important Note**: The version of `dsf-python` in requirements.txt may need to be modified to match your printer's firmware version. The current version (3.6rc2) works with RepRapFirmware 3.6RC2. If you have a different firmware version, please adjust the dsf-python version accordingly in `requirements.txt`.

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

### Vision Tools

The `vision_tools.py` module handles computer vision tasks:

```python
from vision_tools import VisionTools

vision = VisionTools()
vision.start_camera()
```

## Configuration

- Camera parameters can be adjusted in real-time
- HSV thresholds can be tuned for specific components
- Tool offsets are automatically calculated and stored

## Contributing

This is a project for Georgia Tech. Please contact the maintainer for contribution guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Sebastian Mettes  
Georgia Institute of Technology  
Email: smettes3@gatech.edu
