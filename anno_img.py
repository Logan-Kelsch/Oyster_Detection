# anno_img.py

import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO

model = YOLO('active_model/best.pt')  # Load once

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

# Choose your colors and font
BOX_COLOR = (0, 0, 255)  # Red (B, G, R)
TEXT_COLOR = (255, 255, 255)  # White
FONT = cv2.FONT_HERSHEY_COMPLEX  # Choose from list below
FONT_SCALE = 0.6
THICKNESS = 2
TEXT_BG_COLOR = (0, 0, 255)  # Background color for text, optional
import random

def get_class_color(cls_id: int) -> tuple:
    # Generate consistent color based on class ID
    random.seed(cls_id)
    return tuple(random.randint(0, 255) for _ in range(3))


def is_allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def annotate_image_bytes(image_bytes: bytes, ext: str, conf_threshold) -> tuple[BytesIO, str]:

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    results = model(img)[0]

    for box in results.boxes:
        conf = float(box.conf)
        if conf < conf_threshold:
            continue

        cls = int(box.cls)
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = get_class_color(cls)

        label_text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label size
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_x, text_y = x1, y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10

        # Draw label background
        cv2.rectangle(img, (text_x, text_y - text_h - 4), (text_x + text_w, text_y), color, -1)

        # Draw label text
        cv2.putText(img, label_text, (text_x, text_y - 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Encode the image back to bytes
    success, encoded_img = cv2.imencode(f'.{ext}', img)
    if not success:
        raise ValueError("Failed to encode annotated image")

    return BytesIO(encoded_img.tobytes()), f'image/{ext if ext != "jpg" else "jpeg"}'
