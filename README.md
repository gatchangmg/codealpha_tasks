# Real-Time Object Detection and Multi-Object Tracking System

## Overview
This project implements a real-time object detection and multi-object tracking system using YOLOv8 for detection and Deep SORT for tracking. It supports webcam input, video file playback, and synthetic fallback video generation.

## Features
- Real-time detection and bounding box rendering.
- Multi-object tracking with stable track IDs.
- Frame annotation with labels, confidence, object counts, speed estimates, and FPS.
- Optional video recording and frame export.
- CSV logging of tracked object detections.
- Modular design for easy configuration and extension.

## Prerequisites
- Python 3.11 or later
- GPU recommended for performance, but CPU will work

## Setup
1. Create and activate a virtual environment.
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Download or place the YOLO model in `models/yolov8n.pt`.
   - Download model manually from [Ultralytics YOLOv8 releases](https://github.com/ultralytics/ultralytics) or allow the code to load and cache it automatically.

## Running the System
- Use webcam input (default):
  ```bash
  python main.py
  ```
- Use a custom video file:
  1. Set `USE_WEBCAM = False` in `config.py`
  2. Update `VIDEO_INPUT_PATH` with the desired file path
  3. Run:
  ```bash
  python main.py
  ```

## Controls
- `q`: Quit
- `p`: Pause
- `r`: Resume
- `s`: Save current annotated frame
- `Space`: Toggle trail visualization

## Output
- Annotated frames are shown onscreen.
- Recorded video is saved to `outputs/processed_video.mp4` if `RECORD_VIDEO = True`.
- Detection logs are saved to `outputs/detections.csv`.
- Saved frames are stored in `outputs/saved_frames/`.

## Project Structure
- `config.py`: global settings and paths
- `detector.py`: YOLOv8 detection wrapper
- `tracker.py`: Deep SORT tracker wrapper
- `utils.py`: logging, visualization, synthetic video helpers
- `main.py`: pipeline orchestration and display

## Notes
- For best performance, install `torch` with the correct CUDA version for your GPU.
- The default model `yolov8n.pt` is lightweight and suitable for CPU experiments.
- If the webcam returns a black or invalid frame, set `USE_WEBCAM = False` in `config.py` to process a sample or synthetic video instead.
