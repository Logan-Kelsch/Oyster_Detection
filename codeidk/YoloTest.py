from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


def detect_oysters(image_path, model_path='yolov11n-obb.pt'):
    # Load custom YOLO model (trained on oyster data)
    model = YOLO(model_path)

    # Perform detection
    results = model.predict(image_path, conf=0.5)

    # Load image with OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Plot detections
    plt.figure(figsize=(12, 8))

    # Process results
    oyster_count = 0
    for result in results:
        for box in result.boxes:
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            conf = box.conf[0]
            label = f"Oyster: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            oyster_count += 1

    # Display results
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Detected Oysters: {oyster_count}")
    plt.show()

    return oyster_count


# Example usage
if __name__ == "__main__":
    image_path = "path/to/your/oyster_image.jpg"
    detect_oysters(image_path)