from ultralytics import YOLO

#Load trained model
model2 = YOLO("runs/detect/train2/weights/last.pt")

#run inference with model on an image
results = model2("oyestewer.jpg", save=True, show=True)