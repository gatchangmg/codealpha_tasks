import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print('opened', cap.isOpened())
for i in range(3):
    ok, frame = cap.read()
    print('frame', i, ok, frame is not None, frame.shape if frame is not None else None, frame.mean() if frame is not None else None)
    if frame is not None:
        cv2.imshow('frame', frame)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
cap.release()
print('done')
