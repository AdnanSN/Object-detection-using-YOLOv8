# YOLO Object Detection with OpenCV

This project implements an object detection module using the YOLOv8 (You Only Look Once) algorithm and OpenCV. It provides functionalities to detect objects in both images and videos, drawing bounding boxes around detected objects along with their class labels and confidence scores.

## Table of Contents

- [Introduction](#introduction)
- [Machine Learning Concepts](#machine-learning-concepts)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [License](#license)

## Introduction

YOLO is a state-of-the-art, real-time object detection system. It is known for its speed and accuracy, making it suitable for various applications such as autonomous vehicles, surveillance systems, and more. This project utilizes the YOLOv8 model provided by Ultralytics, implemented in Python with OpenCV.

## Machine Learning Concepts

### YOLO (You Only Look Once)

- **Real-Time Object Detection**: YOLO processes images in a single forward pass of the neural network, making it extremely fast.
- **Bounding Boxes**: YOLO predicts bounding boxes and class probabilities directly from full images in one evaluation, unlike traditional approaches that apply classifiers at different locations.
- **Grid Division**: The input image is divided into an SxS grid. Each grid cell is responsible for predicting a certain number of bounding boxes and their respective confidence scores.

### OpenCV

- **Image Processing Library**: OpenCV (Open Source Computer Vision Library) is widely used for real-time computer vision applications. It provides tools for image manipulation, including reading, writing, and displaying images and videos.
- **Integration with YOLO**: OpenCV facilitates the loading of images and videos, drawing bounding boxes, and saving the processed results.

## Project Structure

YOLO-Object-Detection/ 

      ├── detect_objects.py # Main Python script for object detection 
  
      ├── input.jpg # Sample input image (optional) 
  
      ├── input_video.mp4 # Sample input video (optional)
  
      ├── output_image.jpg # Output image after detection 
      
      └── output_video.mp4 # Output video after detection

  
## How It Works

1. **File Upload**: The script prompts the user to upload an image or a video file.
2. **Object Detection**:
   - For images, the `detect_in_image` function detects objects, draws bounding boxes, and saves the output.
   - For videos, the `detect_in_video` function processes each frame, detects objects, and compiles the processed frames into a new video.
3. **Output**: The resulting image or video file is made available for download.

## Requirements

- Python 3.x
- `opencv-python-headless`
- `ultralytics`

## Installation

1. **Clone the Repository**:

       '''
       git clone https://github.com/AdnanSN/Object-detection-using-YOLOv8.git
       cd YOLO-Object-Detection

3. **Install Required Packages**:

You can install the required packages using pip. If you are using Google Colab, you can directly run the following commands in a code cell:

    ```
    !pip install ultralytics opencv-python-headless


## Usage

### Run the Application:

Execute the script using Python:

    ```
    python detect_objects.py

## Upload Files

You will be prompted to upload an image or video file for detection.

## Download Outputs

After processing, the output image or video will be available for download.

## Output

- **Output Image**: An image file (`output_image.jpg`) will be generated with detected objects marked with bounding boxes.
- **Output Video**: A video file (`output_video.mp4`) will be created containing the object detection results for each frame.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


