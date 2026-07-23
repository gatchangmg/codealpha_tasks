import os
import csv
import urllib.request
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

import config

SAMPLE_VIDEO_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/car.mp4"


def download_sample_video():
    if os.path.exists(config.VIDEO_INPUT_PATH):
        return True

    try:
        os.makedirs(os.path.dirname(config.VIDEO_INPUT_PATH), exist_ok=True)
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(SAMPLE_VIDEO_URL, config.VIDEO_INPUT_PATH)
        print(f"[INFO] Sample video downloaded to {config.VIDEO_INPUT_PATH}")
        return True
    except Exception as exc:
        print(f"[WARNING] Sample video download failed: {exc}")
        return False


def generate_synthetic_video():
    if os.path.exists(config.SYNTHETIC_VIDEO_PATH):
        return config.SYNTHETIC_VIDEO_PATH

    os.makedirs(os.path.dirname(config.SYNTHETIC_VIDEO_PATH), exist_ok=True)
    width, height = 1280, 720
    fps = int(config.RECORD_FPS)
    total_frames = fps * 12
    writer = cv2.VideoWriter(config.SYNTHETIC_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_idx in range(total_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (40, 40, 40), 1)
        for y in range(0, height, 80):
            cv2.line(frame, (0, y), (width, y), (40, 40, 40), 1)

        car_x = int((frame_idx * 6) % (width + 200) - 100)
        pedestrian_y = int((frame_idx * 4) % (height + 100) - 50)
        bicycle_x = int(width - (frame_idx * 5) % (width + 120) + 60)
        bicycle_y = int(height - (frame_idx * 3) % (height + 90) + 40)

        cv2.rectangle(frame, (car_x - 70, 450), (car_x + 70, 510), (40, 160, 220), -1)
        cv2.putText(frame, "Car", (car_x - 60, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.circle(frame, (220, pedestrian_y), 30, (220, 80, 80), -1)
        cv2.putText(frame, "Person", (180, pedestrian_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.ellipse(frame, (bicycle_x, bicycle_y), (45, 25), 30, 0, 360, (80, 220, 100), -1)
        cv2.putText(frame, "Bicycle", (bicycle_x - 60, bicycle_y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"[INFO] Synthetic video created at {config.SYNTHETIC_VIDEO_PATH}")
    return config.SYNTHETIC_VIDEO_PATH


def get_simulated_detections(frame_idx, total_frames=360):
    detections = []
    car_x = int((frame_idx * 6) % 1480 - 100)
    pedestrian_y = int((frame_idx * 4) % 820 - 50)
    bicycle_x = int(1280 - (frame_idx * 5) % 1080 - 40)
    bicycle_y = int(720 - (frame_idx * 3) % 620 - 30)

    if -80 < car_x < 1360:
        detections.append({
            "bbox": [car_x - 70, 450, car_x + 70, 510],
            "confidence": 0.88,
            "class_id": 2,
            "class_name": "car",
        })

    if -50 < pedestrian_y < 770:
        detections.append({
            "bbox": [190, pedestrian_y - 30, 250, pedestrian_y + 30],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "person",
        })

    if -50 < bicycle_x < 1320 and -30 < bicycle_y < 750:
        detections.append({
            "bbox": [bicycle_x - 45, bicycle_y - 25, bicycle_x + 45, bicycle_y + 25],
            "confidence": 0.82,
            "class_id": 1,
            "class_name": "bicycle",
        })

    return detections


class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp", "frame", "track_id", "class", "confidence"])

    def log(self, frame_index, track_id, class_name, confidence):
        with open(self.path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                frame_index,
                track_id,
                class_name,
                f"{confidence:.3f}",
            ])

    def summary(self):
        try:
            return pd.read_csv(self.path)
        except Exception:
            return pd.DataFrame()


class LineCrossingCounter:
    def __init__(self, line):
        self.line = line
        self.crossed_in = set()
        self.crossed_out = set()

    def check_crossing(self, track_id, previous, current):
        if previous is None or current is None or self.line is None:
            return None

        if self._intersect(previous, current, self.line[0], self.line[1]):
            if self.line[0][1] == self.line[1][1]:
                direction = "in" if current[1] > previous[1] else "out"
            else:
                direction = "in" if current[0] > previous[0] else "out"

            if direction == "in" and track_id not in self.crossed_in:
                self.crossed_in.add(track_id)
                return "in"
            if direction == "out" and track_id not in self.crossed_out:
                self.crossed_out.add(track_id)
                return "out"

        return None

    @staticmethod
    def _intersect(A, B, C, D):
        def ccw(P, Q, R):
            return (R[1] - P[1]) * (Q[0] - P[0]) > (Q[1] - P[1]) * (R[0] - P[0])
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


class SpeedEstimator:
    def __init__(self, pixels_per_meter, fps):
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.history = {}
        self.speeds = {}

    def estimate(self, track_id, position, frame_index):
        if position is None:
            return 0.0

        if track_id not in self.history:
            self.history[track_id] = (frame_index, position)
            return 0.0

        prev_frame, prev_position = self.history[track_id]
        frame_delta = frame_index - prev_frame
        if frame_delta < config.SPEED_INTERVAL_FRAMES:
            return self.speeds.get(track_id, 0.0)

        dx = position[0] - prev_position[0]
        dy = position[1] - prev_position[1]
        pixel_distance = np.hypot(dx, dy)
        meters = pixel_distance / max(config.PIXELS_PER_METER, 1.0)
        time_sec = max(frame_delta / self.fps, 1e-6)
        speed_mps = meters / time_sec
        speed = speed_mps * 3.6 if config.SPEED_IN_KMH else speed_mps
        self.history[track_id] = (frame_index, position)
        self.speeds[track_id] = speed
        return speed


class Visualizer:
    def __init__(self):
        # Trails per track id
        self.trails = {}
        # Persistent mapping from track_id to a per-class instance number (Person 1, Person 2, ...)
        self.instance_ids = {}
        # Counters to assign the next instance number per class name
        self.class_counters = {}
        # Last known state for tracks (to persist labels when detections drop)
        self.last_seen = {}        # track_id -> last frame index seen
        self.last_bbox = {}        # track_id -> [x1,y1,x2,y2]
        self.last_class = {}       # track_id -> class_name
        self.last_conf = {}        # track_id -> confidence

    @staticmethod
    def get_color(track_id):
        seed = int(hash(str(track_id)) & 0xFFFFFF)
        r = (seed >> 16) & 255
        g = (seed >> 8) & 255
        b = seed & 255
        if r + g + b < 180:
            r = min(r + 80, 255)
            g = min(g + 80, 255)
            b = min(b + 80, 255)
        return (b, g, r)

    def draw(self, frame, tracks, line_counter, speed_estimator, fps, device_name, paused, frame_index):
        output = frame.copy()
        height, width = output.shape[:2]
        class_counts = {}

        # Ensure fps is a non-negative finite number for display
        try:
            fps_display = float(fps)
            if fps_display <= 0 or not np.isfinite(fps_display):
                fps_display = 0.0
        except Exception:
            fps_display = 0.0

        active_ids = set()
        hold_frames = getattr(config, "DETECTION_HOLD_FRAMES", 3)

        # Draw current detections / tracks and update last-known state
        for track in tracks:
            track_id = track["track_id"]
            x1, y1, x2, y2 = map(int, track["bbox"])
            class_name = track["class_name"]
            confidence = track.get("confidence", 0.0)
            color = self.get_color(track_id)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            active_ids.add(track_id)

            # Assign stable per-class instance numbers with re-association logic
            if track_id not in self.instance_ids:
                # Try to find a recently lost track of the same class near this bbox and reuse its instance id
                matched_id = None
                best_dist = float('inf')
                for old_id, last_frame in self.last_seen.items():
                    if last_frame is None:
                        continue
                    # only consider recent ones within hold_frames
                    if frame_index - last_frame > hold_frames:
                        continue
                    if self.last_class.get(old_id) != class_name:
                        continue
                    old_bbox = self.last_bbox.get(old_id)
                    if old_bbox is None:
                        continue
                    cx_new = (x1 + x2) / 2.0
                    cy_new = (y1 + y2) / 2.0
                    cx_old = (old_bbox[0] + old_bbox[2]) / 2.0
                    cy_old = (old_bbox[1] + old_bbox[3]) / 2.0
                    dist = ((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        matched_id = old_id

                # decide threshold based on object size
                if matched_id is not None:
                    old_bbox = self.last_bbox.get(matched_id)
                    diag_new = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    diag_old = ((old_bbox[2] - old_bbox[0]) ** 2 + (old_bbox[3] - old_bbox[1]) ** 2) ** 0.5
                    threshold = max(50.0, max(diag_new, diag_old) * 1.2)
                    if best_dist <= threshold:
                        # reassign instance id and transfer history
                        self.instance_ids[track_id] = self.instance_ids.get(matched_id)
                        # merge/transfer trails
                        if matched_id in self.trails:
                            self.trails.setdefault(track_id, []).extend(self.trails.get(matched_id, []))
                            self.trails[track_id] = self.trails[track_id][-config.TRAIL_MAX_LEN :]
                        # transfer last-known state
                        self.last_seen[track_id] = self.last_seen.get(matched_id)
                        self.last_bbox[track_id] = self.last_bbox.get(matched_id)
                        self.last_class[track_id] = self.last_class.get(matched_id)
                        self.last_conf[track_id] = self.last_conf.get(matched_id)
                        # remove old entries to avoid duplicate display
                        self.last_seen.pop(matched_id, None)
                        self.last_bbox.pop(matched_id, None)
                        self.last_class.pop(matched_id, None)
                        self.last_conf.pop(matched_id, None)
                        self.trails.pop(matched_id, None)
                    else:
                        matched_id = None

                if matched_id is None:
                    next_idx = self.class_counters.get(class_name, 0) + 1
                    self.class_counters[class_name] = next_idx
                    self.instance_ids[track_id] = next_idx

            instance_number = self.instance_ids.get(track_id, 0)

            # Update trails
            self.trails.setdefault(track_id, []).append(center)
            self.trails[track_id] = self.trails[track_id][-config.TRAIL_MAX_LEN :]

            if config.SHOW_TRAILS and len(self.trails[track_id]) > 1:
                for pt1, pt2 in zip(self.trails[track_id][:-1], self.trails[track_id][1:]):
                    cv2.line(output, pt1, pt2, color, 2)

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Label: "ClassName {index} - 91%"
            confidence_pct = min(max(confidence * 100.0, 0.0), 100.0)
            label = f"{class_name.capitalize()} {instance_number} - {confidence_pct:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Make sure label background doesn't go outside image
            label_x1 = max(0, x1)
            label_y1 = max(0, y1 - text_h - 8)
            cv2.rectangle(output, (label_x1, label_y1), (label_x1 + text_w + 8, y1), color, -1)
            cv2.putText(output, label, (label_x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Speed text below box
            speed_value = speed_estimator.speeds.get(track_id, 0.0)
            speed_text = f"{speed_value:.1f} {'km/h' if config.SPEED_IN_KMH else 'm/s'}"
            cv2.putText(output, speed_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Update last-known state
            self.last_seen[track_id] = frame_index
            self.last_bbox[track_id] = [x1, y1, x2, y2]
            self.last_class[track_id] = class_name
            self.last_conf[track_id] = confidence

        # Persist labels for tracks that temporarily lost detection (hold for hold_frames)
        for tid in list(self.last_seen.keys()):
            if tid in active_ids:
                continue
            last_frame = self.last_seen.get(tid, None)
            if last_frame is None:
                continue
            age = frame_index - last_frame
            if age <= hold_frames:
                # draw last-known bbox/label with thinner box to indicate persistence
                bbox = self.last_bbox.get(tid)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                class_name = self.last_class.get(tid, "unknown")
                confidence = self.last_conf.get(tid, 0.0)
                color = self.get_color(tid)

                # faded color for persistence effect (scale toward gray)
                faded = tuple(int(c * 0.6 + 80 * 0.4) for c in color)
                cv2.rectangle(output, (x1, y1), (x2, y2), faded, 1)

                instance_number = self.instance_ids.get(tid, 0)
                confidence_pct = min(max(confidence * 100.0, 0.0), 100.0)
                label = f"{class_name.capitalize()} {instance_number} - {confidence_pct:.0f}%"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                label_x1 = max(0, x1)
                label_y1 = max(0, y1 - text_h - 8)
                cv2.rectangle(output, (label_x1, label_y1), (label_x1 + text_w + 8, y1), faded, -1)
                cv2.putText(output, label, (label_x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                # count persisted for totals
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            else:
                # expired; clean up
                self.last_seen.pop(tid, None)
                self.last_bbox.pop(tid, None)
                self.last_class.pop(tid, None)
                self.last_conf.pop(tid, None)
                self.trails.pop(tid, None)
                # Keep instance_ids so numbering remains stable for the session

        # Draw crossing line if configured
        if config.CROSSING_LINE and config.SHOW_CROSSING_LINE:
            cv2.line(output, config.CROSSING_LINE[0], config.CROSSING_LINE[1], (0, 255, 255), 2)
            cv2.circle(output, config.CROSSING_LINE[0], 5, (0, 255, 255), -1)
            cv2.circle(output, config.CROSSING_LINE[1], 5, (0, 255, 255), -1)

        header = f"Device: {device_name.upper()} | FPS: {fps_display:.1f}"
        if paused:
            header += " | PAUSED"
        cv2.putText(output, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        # Total objects
        if config.SHOW_TOTAL_COUNT:
            total_objects = sum(class_counts.values())
            cv2.putText(output, f"Total Objects: {total_objects}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)

        # Per-class counts
        y = 88
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
            cv2.putText(output, f"{class_name.capitalize()}: {count}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            y += 22

        cv2.putText(output, "Q:Quit  P:Pause  R:Resume  S:Save  SPACE:Toggle", (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        return output
