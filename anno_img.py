# anno_img.py
import cv2
from ultralytics import YOLO
import os


def run(in_p,out_p):
    
    #load in the model that will be used, will keep this location constant
    model = YOLO('active_model/best.pt')
    
    #load in the image using cv2
    img = cv2.imread(in_p)
    #ensure validity of image file
    #cv2 returns None if it cannot be done
    if img is None:
        raise FileNotFoundError(f"Cannot read image at {in_p}")

    #model inference called here through ultralytics functionality
    results = model(img)
    
    #new image is generated with annotations
    annotated_img = results[0].plot()
    
    #save this file into the repository
    cv2.imwrite(out_p, annotated_img)
    return out_p

if __name__ == "__main__":
    import sys
    import cv2
    from ultralytics import YOLO
    import os
    in_p, out_p = sys.argv[1], sys.argv[2]
    run(in_p,out_p)
    