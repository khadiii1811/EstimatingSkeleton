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


<h2>This is list of used libraries</h2>
We used CNN, cvzone/mediapipe, OpenCV (CV2), tkinter ....

![New Bitmap Image](https://user-images.githubusercontent.com/59195029/204079521-512acc7c-fb96-4e3f-99c6-b5a2cbd7a9ca.jpg)

<h2>Program Interface Using Tkinter</h2>

![New Bitmap Image_2](https://user-images.githubusercontent.com/59195029/204079964-35a3d30c-1949-4d7e-b7d1-f6da9abeb2d8.jpg)


<h2>Results</h2>

![New Bitmap Image_3](https://user-images.githubusercontent.com/59195029/204079884-a6c08434-6b44-4c98-a67f-b11bd4e1faa0.jpg)

![New Bitmap Image_4](https://user-images.githubusercontent.com/59195029/204079885-ba9928fc-a32a-4f3c-92ad-33a57ba6bf2d.jpg)

![New Bitmap Image_4_2](https://user-images.githubusercontent.com/59195029/204080082-0c11a150-697a-4e64-ae19-24d732bd898d.jpg)

![New Bitmap Image_4_3](https://user-images.githubusercontent.com/59195029/204080088-8c4f801f-a741-4a42-b279-cf42ba2c9a46.jpg)

![New Bitmap Image_4_4](https://user-images.githubusercontent.com/59195029/204080094-a20cc2e5-2d85-4697-9059-db980e6efcdd.jpg)

![New Bitmap Image_4_5](https://user-images.githubusercontent.com/59195029/204080099-f02b3a62-5835-4cab-aaf6-76cee4db3e74.jpg)

![New Bitmap Image_4_5_2](https://user-images.githubusercontent.com/59195029/204080104-8a4d9328-b51a-46c8-b619-5b4ec5bb027e.jpg)

![New Bitmap Image_4_5_3](https://user-images.githubusercontent.com/59195029/204080108-eed35c26-b187-4959-99b9-74ea2b74ea3e.jpg)
