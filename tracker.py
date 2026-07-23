import config
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    """Wraps Deep SORT for multi-object tracking with consistent IDs."""

    def __init__(self):
        print("[INFO] Initializing Deep SORT tracker...")
        self.tracker = DeepSort(
            max_age=config.MAX_AGE,
            n_init=config.N_INIT,
            max_cosine_distance=config.MAX_COSINE_DISTANCE,
            nn_budget=config.NN_BUDGET,
            embedder="mobilenet",
            half=(config.DEVICE == "cuda"),
            bgr=True,
        )

    def update(self, detections, frame):
        deepsort_dets = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            width = x2 - x1
            height = y2 - y1
            deepsort_dets.append(([x1, y1, width, height], det["confidence"], det["class_name"]))

        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        track_list = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update != 0:
                continue

            bbox = track.to_tlbr()
            class_name = track.get_det_class() if hasattr(track, "get_det_class") else getattr(track, "det_class", "unknown")
            confidence = track.get_det_conf() if hasattr(track, "get_det_conf") else float(getattr(track, "det_conf", 1.0))

            track_list.append({
                "track_id": track.track_id,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "class_name": class_name,
                "confidence": float(confidence),
            })

        return track_list
