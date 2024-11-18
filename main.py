from ultralytics import YOLO 
import numpy as np
import torch

if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")
    results = model.train(data="config.yaml", epochs=50, batch=32, device=0)




