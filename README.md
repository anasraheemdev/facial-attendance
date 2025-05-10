# Facial Attendance System

A computer vision-based attendance management system that uses facial recognition to automate the process of recording attendance.

![Facial Attendance System](https://miro.medium.com/v2/resize:fit:4800/1*-IDdJP45FHWLq4VZxKPKYw.png)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Architecture](#technical-architecture)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Facial Attendance System is a Python application that leverages OpenCV and computer vision techniques to create a secure, reliable, and efficient attendance tracking solution using facial recognition technology. The system captures facial images through a webcam, processes them to identify registered users, and maintains accurate attendance records.

## Features

- **User Registration**: Register new users by capturing multiple face samples for reliable recognition
- **Facial Recognition**: Identify registered users through advanced face recognition algorithms
- **Attendance Tracking**: Automatically record attendance with time stamps
- **View Attendance Records**: Display and browse historical attendance data
- **Recognition Sensitivity**: Adjustable recognition threshold to balance security and usability
- **Face Enhancement**: Built-in techniques to improve recognition accuracy:
  - Contrast enhancement
  - Noise reduction
  - Image sharpening
  - Adaptive thresholding

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy
- Webcam or camera device

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anasraheemdev/facial-attendance-system.git
   cd facial-attendance-system
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script to start the application:

```bash
python main.py
```

### Main Menu

The system presents a simple menu interface with the following options:

1. **Mark Attendance (Existing User)**: Activate the camera to recognize and record attendance for registered users.
2. **Register New User**: Add a new user to the system by capturing face samples.
3. **View Attendance Records**: Browse attendance data by date.
4. **Adjust Recognition Sensitivity**: Fine-tune the facial recognition threshold.
5. **Improve Face**: Demo of various face enhancement techniques.
6. **Exit**: Close the application.

### User Registration Process

When registering a new user:
1. Enter your full name when prompted
2. Look at the camera and press 'c' to capture face samples (5 samples recommended)
3. Samples will be processed and stored securely for future recognition

### Marking Attendance

To mark attendance:
1. Select "Mark Attendance" from the menu
2. Position your face in front of the camera
3. The system will attempt to recognize your face
4. Once recognized, your attendance will be recorded automatically
5. Press 'q' to return to the main menu

## Technical Architecture

The system uses an object-oriented design with the following key components:

- **Face Detection**: Uses Haar Cascade classifier to detect faces in camera frames
- **Face Recognition**: Employs multiple advanced techniques for robust recognition:
  - SIFT feature extraction and matching
  - Histogram comparison
  - Local Binary Pattern analysis
  - Structural similarity index
- **Face Enhancement**: Various preprocessing techniques to improve image quality
- **Data Management**: Simple file-based storage for user data and attendance records

### Directory Structure

```
facial-attendance-system/
├── data/
│   ├── users/           # Stores registered user face templates
│   └── attendance/      # Contains attendance records by date
├── main.py              # Main application code
└── requirements.txt     # Project dependencies
```

## Advanced Features

### Multiple Recognition Metrics

The system combines multiple face recognition methods to achieve high accuracy:

- **Histogram Correlation**: Compares intensity distributions
- **Local Binary Patterns**: Analyzes texture patterns
- **SIFT Feature Matching**: Scale-invariant feature detection
- **Structural Similarity**: Evaluates structural differences

### Face Enhancement Techniques

To improve recognition in challenging conditions, the system implements:

- **Histogram Equalization**: Improves contrast in poor lighting
- **Gaussian Blur**: Reduces image noise
- **Unsharp Masking**: Enhances edge definition
- **Adaptive Thresholding**: Improves feature extraction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Created with ❤️ by Anas Raheem
