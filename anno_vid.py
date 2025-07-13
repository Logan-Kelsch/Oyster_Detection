# anno_vid.py

import cv2
import os
import numpy as np
from ultralytics import YOLO
from io import BytesIO
import tempfile

model = YOLO('active_model/best.pt')  # load once

def annotate_video_bytes(video_bytes: bytes, output_format='mp4') -> tuple[BytesIO, str]:
    """
    Annotate a video (raw bytes) with YOLO and return a BytesIO stream + mime type.
    """
    # 1️⃣ Write input bytes to a temp file
    fd_in, path_in = tempfile.mkstemp(suffix='.mp4')
    os.close(fd_in)
    with open(path_in, 'wb') as f:
        f.write(video_bytes)

    # 2️⃣ Open with OpenCV
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        os.remove(path_in)
        raise IOError("Could not open video from memory buffer")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3️⃣ Prepare temp output file
    fd_out, path_out = tempfile.mkstemp(suffix=f'.{output_format}')
    os.close(fd_out)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path_out, fourcc, fps, (width, height))

    # 4️⃣ Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        writer.write(annotated_frame)

    cap.release()
    writer.release()

    # 5️⃣ Read output back into memory
    with open(path_out, 'rb') as f:
        annotated_bytes = f.read()

    # 6️⃣ Cleanup temp files
    os.remove(path_in)
    os.remove(path_out)

    return BytesIO(annotated_bytes), 'video/mp4'
