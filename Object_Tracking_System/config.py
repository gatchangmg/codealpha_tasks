import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SAVED_FRAMES_DIR = os.path.join(OUTPUT_DIR, "saved_frames")

for folder in [MODELS_DIR, VIDEOS_DIR, OUTPUT_DIR, SAVED_FRAMES_DIR]:
    os.makedirs(folder, exist_ok=True)

MODEL_NAME = "yolov8n.pt"  # Use the smallest YOLOv8 model for CPU-friendly real-time inference
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMG_SIZE = 416
MAX_DETECTIONS = 100
AUGMENT = False  # Disable expensive multi-scale augmentation on CPU to avoid frozen frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES_TO_DETECT = None

USE_WEBCAM = True
CAMERA_INDEX = 0
VIDEO_INPUT_PATH = os.path.join(VIDEOS_DIR, "sample.mp4")
SYNTHETIC_VIDEO_PATH = os.path.join(VIDEOS_DIR, "synthetic_sample.mp4")
GENERATE_SYNTHETIC_FALLBACK = True

MAX_AGE = 30
N_INIT = 3
MAX_COSINE_DISTANCE = 0.2
NN_BUDGET = 100

CSV_LOG_PATH = os.path.join(OUTPUT_DIR, "detections.csv")
RECORD_VIDEO = True
RECORD_PATH = os.path.join(OUTPUT_DIR, "processed_video.mp4")
RECORD_FPS = 30.0
DETECTION_HOLD_FRAMES = 3

SHOW_ANNOTATIONS = True
ROI_RECT = None
SHOW_ROI = False
CROSSING_LINE = None
SHOW_CROSSING_LINE = False
SHOW_TRAILS = True
TRAIL_MAX_LEN = 25
PIXELS_PER_METER = 15.0
SPEED_INTERVAL_FRAMES = 10
SPEED_IN_KMH = True
SHOW_FPS = True
SHOW_DEVICE = True
SHOW_TOTAL_COUNT = True
