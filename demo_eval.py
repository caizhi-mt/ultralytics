import torch
import torch_musa
from ultralytics import YOLO

musa_model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
print(musa_model("https://ultralytics.com/images/bus.jpg"))
