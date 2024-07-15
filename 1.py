import ultralytics
from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.train(data="D:\shivu\people\data.yaml", epochs =3)