import os
import cv2
import numpy as np
from ultralytics import YOLO
import config

class YOLODetector:
    """Wraps the Ultralytics YOLOv8 model for single-frame detection."""

    def __init__(self):
        print(f"[INFO] Initializing YOLOv8 on device: {config.DEVICE}")
        model_source = config.MODEL_PATH if os.path.exists(config.MODEL_PATH) else config.MODEL_NAME
        self.model = YOLO(model_source)
        print(f"[INFO] YOLO model loaded from {model_source}")
        self.class_names = self.model.names

    def detect(self, frame):
        if frame is None:
            return []

        results = self.model(
            source=frame,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=config.DEVICE,
            imgsz=config.IMG_SIZE,
            max_det=config.MAX_DETECTIONS,
            augment=config.AUGMENT,
            verbose=False,
        )

        detections = []
        if len(results) == 0:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names.get(class_id, "unknown")

            if config.CLASSES_TO_DETECT is not None and class_id not in config.CLASSES_TO_DETECT:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
            })

        return detections
