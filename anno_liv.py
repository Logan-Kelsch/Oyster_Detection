
# anno_liv.py
import cv2
# … load YOLO model …

def frame_generator():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # … detect & annotate on frame …
        _, jpeg = cv2.imencode('.jpg', frame)
        yield jpeg.tobytes()
    cap.release()
