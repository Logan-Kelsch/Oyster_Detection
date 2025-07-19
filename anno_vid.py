# anno_vid.py

import cv2
import os
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import tempfile

# load once
model = YOLO('active_model/best.pt')
model_fps = 5.0

# precompute a distinct color for each class
def make_color_palette(n):
    """Generate n visually distinct BGR colors via HSV spacing."""
    palette = []
    for i in range(n):
        hue = int(179 * i / n)
        hsv = np.uint8([[[hue, 255, 255]]])           # full saturation/value
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette

CLASS_PALETTE = make_color_palette(len(model.names))

def choose_text_color(bgr):
    """Pick black or white text for max contrast against bgr bg."""
    # brightness = (B + G + R) / 3
    return (0, 0, 0) if sum(bgr)/3 > 128 else (255, 255, 255)

def annotate_video_bytes(video_bytes: bytes, output_format='mp4') -> tuple[BytesIO, str]:
    # 1) write input→tmp
    fd_in, path_in = tempfile.mkstemp(suffix='.mp4'); os.close(fd_in)
    with open(path_in, 'wb') as f: f.write(video_bytes)

    # 2) open capture
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        os.remove(path_in)
        raise IOError("Could not open video")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    skip = max(1, int(round(video_fps / model_fps)))

    # 3) prepare writer
    fd_out, path_out = tempfile.mkstemp(suffix=f'.{output_format}'); os.close(fd_out)
    writer = cv2.VideoWriter(
        path_out, cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps, (w, h)
    )

    last_boxes = None
    frame_id = 0

    FONT           = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE     = 0.5
    FONT_THICKNESS = 1
    PAD            = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inference every skip frames
        if frame_id % skip == 0:
            results = model(frame)[0]
            last_boxes = [
                (
                    int(x1), int(y1), int(x2), int(y2),
                    int(cls), float(conf)
                )
                for (x1, y1, x2, y2), cls, conf
                in zip(results.boxes.xyxy,
                       results.boxes.cls,
                       results.boxes.conf)
            ]

        out = frame.copy()
        if last_boxes:
            for xmin, ymin, xmax, ymax, cls_id, conf in last_boxes:
                # pick class color
                color = CLASS_PALETTE[cls_id]
                text_color = choose_text_color(color)

                # 1) draw box border
                cv2.rectangle(out, (xmin, ymin), (xmax, ymax),
                              color, thickness=2)

                # 2) label
                label = f"{model.names[cls_id]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, FONT,
                                              FONT_SCALE, FONT_THICKNESS)
                y0 = max(0, ymin - th - 2 * PAD)
                y1 = ymin
                x1 = xmin + tw + 2 * PAD
                # label background
                cv2.rectangle(out,
                              (xmin, y0),
                              (x1, y1),
                              color, -1)
                # label text
                cv2.putText(out, label,
                            (xmin + PAD, y1 - PAD),
                            FONT, FONT_SCALE,
                            text_color, FONT_THICKNESS)

        writer.write(out)
        frame_id += 1

    cap.release()
    writer.release()

    # 5) read back & cleanup
    with open(path_out, 'rb') as f: annotated_bytes = f.read()
    os.remove(path_in); os.remove(path_out)
    return BytesIO(annotated_bytes), 'video/mp4'