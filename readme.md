# Robotik-B

A collection of robotics projects featuring computer vision and autonomous navigation systems.

## Table of Contents

- [Projects](#projects)
  - [BNU-V2-with-EfficientDet](#bnu-v2-with-efficientdet)
    - [Features](#features)
    - [Hardware Requirements](#hardware-requirements)
    - [Software Requirements](#software-requirements)
    - [Setup](#setup)
    - [Usage](#usage)
    - [Robot Behavior](#robot-behavior)
    - [Project Structure](#project-structure)
    - [Technical Details](#technical-details)
  - [UTS-efficientDet-Demo](#uts-efficientdet-demo)
    - [Features](#features-1)
    - [Project Structure](#project-structure-1)
    - [Usage](#usage-1)
    - [Technical Details](#technical-details-1)
    - [Documentation](#documentation)
- [License](#license)
- [Contributing](#contributing)
- [Academic Context](#academic-context)

## Projects

### BNU-V2-with-EfficientDet

An intelligent robot surveillance system that combines real-time object detection with autonomous navigation. The robot can detect specific objects and respond accordingly by stopping or continuing its search pattern.

#### Features

- **Real-time Object Detection**: Uses EfficientDet Lite model for fast and accurate object detection
- **Autonomous Navigation**: Robot can move forward, backward, turn left/right, and stop
- **Target Object Recognition**: Specifically detects bottles, persons, and chairs
- **Serial Communication**: Arduino-Python communication for motor control
- **Live Camera Feed**: Real-time video processing with bounding box visualization

#### Hardware Requirements

- ESP32 or Arduino-compatible microcontroller
- 2x DC motors with motor driver
- USB camera
- Serial connection (USB cable)
- Motor pins configuration:
  - Left Motor: pins 22, 23 (direction), pin 21 (PWM)
  - Right Motor: pins 17, 18 (direction), pin 16 (PWM)

#### Software Requirements

```bash
# Python dependencies
pip install opencv-python
pip install numpy
pip install tflite-runtime
pip install pyserial
```

#### Setup

1. **Arduino Setup**:
   - Upload `bnu5.ino` to your microcontroller
   - Connect motors according to pin configuration
   - Ensure serial communication on port 115200 baud

2. **Python Setup**:
   - Download EfficientDet Lite model to specified path
   - Adjust camera device index in `cv2.VideoCapture(0)`
   - Update serial port in the Python script (`/dev/ttyUSB0` for Linux)

3. **Model Setup**:
   - Place `efficientdet_lite0.tflite` model file in the specified directory
   - Update `MODEL_PATH` variable if needed

#### Usage

1. Start the Arduino with uploaded firmware
2. Run the Python detection script:
   ```bash
   python3 real_time_object_detection_asli.py
   ```
3. The robot will start searching by rotating left
4. When target objects (bottle, person, chair) are detected, the robot stops
5. Press 'q' to quit the application

#### Robot Behavior

- **Search Mode**: Robot rotates left continuously when no target objects are detected
- **Detection Mode**: Robot stops when bottle, person, or chair is detected
- **Manual Control**: Send commands via serial:
  - '1': Move forward
  - '2': Turn right  
  - '3': Turn left
  - '4': Move backward
  - '0': Stop

#### Project Structure

```
BNU-V2-with-EfficientDet/
├── bnu5.ino                           # Arduino motor control firmware
├── real_time_object_detection_asli.py # Python object detection script
└── efficientdet_lite0.tflite         # TensorFlow Lite model (download required)
```

#### Technical Details

- **Object Detection**: EfficientDet Lite0 model with COCO dataset (91 classes)
- **Detection Threshold**: 0.2 probability threshold
- **PWM Frequency**: 30kHz for motor control
- **Communication**: 115200 baud serial communication
- **Video Processing**: Real-time frame processing with OpenCV

### UTS-efficientDet-Demo

A demonstration project showcasing EfficientDet object detection implementation. This project appears to be part of academic coursework (UTS - Ujian Tengah Semester) and includes documentation for PKM-KC (Program Kreativitas Mahasiswa - Karsa Cipta) 2025 proposal.

#### Features

- **EfficientDet D0 Model**: Implementation using EfficientDet D0 architecture
- **White Blood Cell Detection**: Specialized detection for medical/biological applications (`effdet_wbc.py`)
- **Academic Documentation**: Includes progress reports and research proposals

#### Project Structure

```
UTS-efficientDet-Demo/
├── Progres ROBOTIK Tim A.pdf          # Team A robotics progress report
├── Proposal PKM-KC 2025 EfficientDet-D1.docx # PKM-KC 2025 proposal document
└── effdet-demo/
    ├── .gitignore
    ├── effdet_wbc.py                  # White Blood Cell detection script
    └── models/
        └── efficientdet_d0/           # EfficientDet D0 saved model
            ├── fingerprint.pb
            ├── saved_model.pb
            └── variables/
                ├── variables.data-00000-of-00001
                └── variables.index
```

#### Usage

```bash
# Navigate to the demo directory
cd UTS-efficientDet-Demo/effdet-demo/

# Run the white blood cell detection demo
python effdet_wbc.py
```

#### Technical Details

- **Model Architecture**: EfficientDet D0 (saved model format)
- **Application Domain**: Medical imaging / White Blood Cell analysis
- **Framework**: TensorFlow/Keras saved model format
- **Academic Context**: Part of PKM-KC 2025 research proposal

#### Documentation

- **Progress Report**: [`Progres ROBOTIK Tim A.pdf`](UTS-efficientDet-Demo/Progres%20ROBOTIK%20Tim%20A.pdf)
- **Research Proposal**: [`Proposal PKM-KC 2025 EfficientDet-D1.docx`](UTS-efficientDet-Demo/Proposal%20PKM-KC%202025%20EfficientDet-D1.docx)

## License

This project is open source. Please check individual project directories for specific license information.

## Contributing

Feel free to contribute to any of the projects by submitting pull requests or reporting issues.

## Academic Context

This repository contains projects developed as part of academic research and coursework, including:
- Robotics course assignments (UTS projects)
- PKM-KC (Program Kreativitas Mahasiswa - Karsa Cipta) 2025 research proposals
- Team-based robotics development initiatives

For academic inquiries or collaboration opportunities, please refer to the documentation in individual project directories.