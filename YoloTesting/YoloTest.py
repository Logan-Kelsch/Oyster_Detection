from ultralytics import YOLO

#Load a model
model = YOLO("yolo11n.pt")

#Train the model
#results = model.train(data='RoboFlowOysterData/data.yaml', epochs=100, imgsz=640)
model2 = YOLO("runs/detect/train2/weights/best.pt")

#run inference with model on an image
results = model2("oyestewer.jpg", save=True, show=True)