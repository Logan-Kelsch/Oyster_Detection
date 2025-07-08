# anno_vid.py

import sys
import cv2

def run(input_path: str, output_path: str) -> str:
    """
    Reads a video from `input_path`, applies YOLO-based annotations frame by frame,
    and writes the result to `output_path`.
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {input_path}")

    # Gather input video properties
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output writer (MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== YOLO ANNOTATION PLACEHOLDER =====
        # Here you would run your model on `frame` and draw boxes, e.g.:
        # detections = yolo_model.detect(frame)
        # for det in detections:
        #     x1, y1, x2, y2, label, conf = det
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        #     cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        # ========================================

        # For now, just pass the unmodified frame through:
        annotated_frame = frame

        writer.write(annotated_frame)

    # Clean up
    cap.release()
    writer.release()

    return output_path

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python anno_vid.py <input_video_path> <output_video_path>")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    result = run(inp, outp)
    print(f"Annotated video saved to {result}")
