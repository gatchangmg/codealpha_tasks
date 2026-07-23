import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

import config
from detector import YOLODetector
from tracker import ObjectTracker
from utils import CSVLogger, Visualizer, SpeedEstimator, LineCrossingCounter, download_sample_video, generate_synthetic_video, get_simulated_detections


def is_valid_frame(frame):
    if frame is None or frame.size == 0:
        return False
    if frame.min() == frame.max() == 0:
        return False
    if frame.mean() <= 5:
        return False
    return True


def open_camera(index=0):
    backends = [
        (cv2.CAP_DSHOW, "CAP_DSHOW"),
        (cv2.CAP_MSMF, "CAP_MSMF"),
        (cv2.CAP_ANY, "CAP_ANY"),
    ]
    codecs = [
        ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
        ("YUY2", cv2.VideoWriter_fourcc(*"YUY2")),
        (None, None),
    ]

    for backend, name in backends:
        for codec_name, codec in codecs:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            if codec is not None:
                capture.set(cv2.CAP_PROP_FOURCC, codec)

            time.sleep(0.2)
            valid_frame = False
            for attempt in range(10):
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                if is_valid_frame(frame):
                    valid_frame = True
                    break

            if valid_frame:
                backend_label = f"{name}{'+' + codec_name if codec_name else ''}"
                print(f"[INFO] Opened webcam with backend {backend_label}")
                return capture, backend_label

            capture.release()
            print(f"[WARN] Webcam opened with {name} {codec_name or 'default'} but failed to return a valid frame.")

    return None, None


def select_video_source():
    if config.USE_WEBCAM:
        capture, backend_name = open_camera(config.CAMERA_INDEX)
        if capture is not None:
            return capture, f"webcam ({backend_name})"

        print("[WARN] Webcam could not be initialized. Falling back to video input.")

    if os.path.exists(config.VIDEO_INPUT_PATH):
        return cv2.VideoCapture(config.VIDEO_INPUT_PATH), os.path.basename(config.VIDEO_INPUT_PATH)

    if download_sample_video():
        return cv2.VideoCapture(config.VIDEO_INPUT_PATH), os.path.basename(config.VIDEO_INPUT_PATH)

    if config.GENERATE_SYNTHETIC_FALLBACK:
        synthetic_path = generate_synthetic_video()
        return cv2.VideoCapture(synthetic_path), os.path.basename(synthetic_path)

    raise RuntimeError("Unable to initialize any video source.")


def configure_video_writer(frame_width, frame_height):
    if not config.RECORD_VIDEO:
        return None

    os.makedirs(os.path.dirname(config.RECORD_PATH), exist_ok=True)
    writer = cv2.VideoWriter(
        config.RECORD_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.RECORD_FPS,
        (frame_width, frame_height),
    )
    return writer


def main():
    capture, source_name = select_video_source()
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source_name}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or config.RECORD_FPS

    detector = YOLODetector()
    tracker = ObjectTracker()
    logger = CSVLogger(config.CSV_LOG_PATH)
    visualizer = Visualizer()
    speed_estimator = SpeedEstimator(config.PIXELS_PER_METER, fps)
    line_counter = LineCrossingCounter(config.CROSSING_LINE)

    video_writer = configure_video_writer(frame_width, frame_height)
    inference_executor = ThreadPoolExecutor(max_workers=1)
    inference_future = None
    frame_index = 0
    paused = False
    source_label = source_name

    print(f"[INFO] Starting object tracking pipeline for {source_label}")
    cv2.namedWindow("Object Detection & Tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("Object Detection & Tracking", frame_width, frame_height)

    while True:
        if not paused:
            has_frame, frame = capture.read()
            if not has_frame or frame is None:
                print("[INFO] End of stream or camera disconnected.")
                break

            if frame.size == 0 or frame.mean() == 0:
                print("[WARN] Received invalid or empty frame from source.")
                break

            if inference_future is None:
                inference_future = inference_executor.submit(detector.detect, frame.copy())

            if inference_future.done():
                detections = inference_future.result()
                inference_future = None
            else:
                detections = []

            if detections:
                tracks = tracker.update(detections, frame)
            else:
                tracks = []

            for track in tracks:
                logger.log(frame_index, track["track_id"], track["class_name"], track["confidence"])

            output = visualizer.draw(frame, tracks, line_counter, speed_estimator, fps, config.DEVICE, paused, frame_index)
            frame_index += 1

            if video_writer is not None:
                video_writer.write(output)
        else:
            output = visualizer.draw(frame, tracks if 'tracks' in locals() else [], line_counter, speed_estimator, fps, config.DEVICE, paused, frame_index)

        if output is None or output.size == 0:
            print("[ERROR] Visualization produced empty output frame.")
            break

        cv2.imshow("Object Detection & Tracking", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("p"):
            paused = True
        if key == ord("r"):
            paused = False
        if key == ord("s"):
            save_path = os.path.join(config.SAVED_FRAMES_DIR, f"frame_{frame_index:05d}.jpg")
            cv2.imwrite(save_path, output)
            print(f"[INFO] Saved frame to {save_path}")
        if key == 32:
            config.SHOW_TRAILS = not config.SHOW_TRAILS
            print(f"[INFO] SHOW_TRAILS set to {config.SHOW_TRAILS}")

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline terminated gracefully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
