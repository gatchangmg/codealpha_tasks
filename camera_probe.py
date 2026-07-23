import cv2
import time

BACKENDS = [
    (cv2.CAP_DSHOW, 'CAP_DSHOW'),
    (cv2.CAP_MSMF, 'CAP_MSMF'),
    (cv2.CAP_ANY, 'CAP_ANY'),
]

for index in range(3):
    print(f'=== Testing camera index {index} ===')
    for backend, name in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        print(f'backend={name}, opened={cap.isOpened()}')
        if not cap.isOpened():
            cap.release()
            continue

        print('  width', cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'height', cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 'convert_rgb', cap.get(cv2.CAP_PROP_CONVERT_RGB))
        for i in range(3):
            ok, frame = cap.read()
            print(f'  frame {i} ok={ok} shape={None if frame is None else frame.shape} mean={None if frame is None else frame.mean()} min={None if frame is None else frame.min()} max={None if frame is None else frame.max()}')
        cap.release()
    print()

print('=== Done ===')
