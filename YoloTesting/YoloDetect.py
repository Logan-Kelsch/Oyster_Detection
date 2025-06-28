from ultralytics import YOLO

#Load trained model
model2 = YOLO("runs/detect/train3/weights/best.pt")

#run inference with model on an image
results = model2("dogsInPark.jpg", save=True, show=True)