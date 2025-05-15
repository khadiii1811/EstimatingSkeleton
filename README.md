# Human Skeleton Estimating Application

A computer vision application for human skeleton estimation, fall detection, face recognition, and performance evaluation.

## Project Structure

The application has been modularized into multiple Python files for better organization and maintainability:

- **main.py**: Entry point that initializes the application and sets up the UI
- **detectors.py**: Classes for different detection techniques (PoseDetector, etc.)
- **video_processor.py**: Functions for processing video streams and calculating metrics
- **evaluation.py**: Performance evaluation and reporting functions
- **ui_components.py**: UI creation and styling functions
- **utils.py**: Utility functions for various parts of the application

## Features

- Real-time pose estimation from webcam
- Image pose estimation from files
- Face detection and tracking
- Hand tracking
- Face mesh detection
- Fall detection (with alarm)
- Performance evaluation system with metrics for:
  - Performance (FPS, stability)
  - Accuracy (detection rates, confidence)
  - Resource usage (CPU, memory)
  - Usability (response time)
- Performance visualization and reporting (PDF/Excel exports)

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- CVZone
- TKinter
- PIL (Pillow)
- ReportLab (for PDF generation)
- Openpyxl (for Excel export)
- Psutil (for resource monitoring)

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install opencv-python mediapipe cvzone pillow reportlab openpyxl psutil
   ```
3. Run the application:
   ```
   python main.py
   ```

## Usage

The application has a user-friendly interface with two main tabs:

1. **Skeleton Estimation**: Basic pose estimation functionality
2. **Extended Functions**: Additional features like face detection, hand tracking, fall detection, and performance evaluation

The status bar at the bottom shows the current state of the application.

## Performance Evaluation

The application includes a comprehensive evaluation system that analyzes:

- FPS and stability
- Detection accuracy and confidence
- CPU and memory usage
- Response time and user experience

Results can be viewed in the application and exported to PDF or Excel format.

## Team Members

- Nguyễn Nguyên Toàn Khoa (22110044)
- Nguyen Hoang Huy (22110036)
- Nguyễn Lê Tùng Chi (22110013)
- Nguyen Thành Tính (22110077)

# Human-Skeleton-Estimating
 Application Human Skeleton Estimating using image processing and pattern recognition
Video demo link: https://youtu.be/DQVGjCHT1J4
<br>This video is my Final Project (Image Processing Course) about: HUMAN SKELETON ESTIMATING FROM IMAGE (VIDEO) USING IMAGE PROCESSING AND PATTERN RECOGNITION TECHNIQUES <br>

﻿# About project

<h4>In this project we will use Image processing which performs some operations on an image, in order to get the human skeleton estimation from the image or frame (from video). But before that, we need to apply Pattern recognition so that the app will automatically detect human parts by using machine learning algorithms.</h4>



