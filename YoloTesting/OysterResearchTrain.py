import torch
import os
from IPython import display
from IPython.display import display, Image
from ultralytics import YOLO
from roboflow import Roboflow
import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

HOME =  os.getcwd()
print(HOME)

#train many smaller models:
#this was trained on a RTX 4080.
#I trained one model at a time over the course of several hours, you may train multiple
#in a row if you have the time and watch a computer for a long time


print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")

models = ["yolov8m.pt"] #"yolov8n.pt",

#image size is 960 for everything except medium, for gpu mem alloc issues
for m in models:
    model = YOLO(m)
    model.train(
        data=dataset.location + "/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        patience=20,
        amp=True,
        optimize=True, #amp=True, cache=False
        project="oysterTrainedModels",
        name=f"model_{m.split('.')[0]}",
        exist_ok=True
    )
