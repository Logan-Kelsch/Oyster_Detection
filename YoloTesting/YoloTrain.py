# This program trains a basic yolo model into our own custom model after its been trained
from ultralytics import YOLO


#Load a model
model = YOLO("yolo11n.pt")


#Train the model
results = model.train(
    data='RoboFlowOysterData/data.yaml',
    epochs=100,  # Number of complete dataset passes
    patience=10,  # Stop if no improvement for 10 epochs (Too many epochs without validation can be bad)
    batch=16, #  Number of images proccessed before updating its internal weights.
    #imgsz=640
)
