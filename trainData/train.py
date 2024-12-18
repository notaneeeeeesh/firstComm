from ultralytics import YOLO 
import numpy as np
import torch

epochs=80
batch=0.90
device=0
patience=10
if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")
    results = model.train(data="config.yaml", epochs=80, batch=0.90, device=0, patience=10)
