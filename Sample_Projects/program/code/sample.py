from ultralytics import YOLO
import cv2

model = YOLO("best_v.pt")
results = model.predict("chassis.jpg", conf=0.25, save=True)
