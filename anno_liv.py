# anno_liv.py
import cv2
import numpy as np
from ultralytics import YOLO
import base64

# load once
model = YOLO("active_model/best.pt")
model_fps = 5.0    # you can expose this via an endpoint if you prefer

def detect_objects_from_bytes(frame_bytes: bytes, conf_threshold=0.75):
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = model(frame)
    detections = []

    for box in results[0].boxes:
        if float(box.conf) >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf  = float(box.conf)
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "label": label, "conf": conf
            })

    return detections
