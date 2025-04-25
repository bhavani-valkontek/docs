import cv2
import easyocr
import torch
from ultralytics import YOLO
import numpy as np
import time
import os
import json
import csv

# --- CONFIG ---
MODEL_PATH = "best_news.pt"                # Your YOLO model
IMAGE_PATH = "tophat.jpg"
OUTPUT_FOLDER = "emb_output"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 640

# --- PREP ---
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
model = YOLO(MODEL_PATH)
image = cv2.imread(IMAGE_PATH)
image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

# --- YOLO DETECTION ---
start = time.time()
results = model(image_resized)[0]
end = time.time()

# --- FILTER FOR TOP CONFIDENCE DETECTION ---
boxes = results.boxes
if len(boxes) == 0:
    print("No detections found.")
    exit()

# Get the box with highest confidence
boxes = sorted(boxes, key=lambda b: b.conf[0], reverse=True)
top_box = boxes[0]

x1, y1, x2, y2 = map(int, top_box.xyxy[0])
conf = top_box.conf[0].item()
if conf < CONFIDENCE_THRESHOLD:
    print("Top box below confidence threshold.")
    exit()

# --- CROP DETECTED REGION ---
plate_crop = image_resized[y1:y2, x1:x2]
gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- OCR ---
reader = easyocr.Reader(['en'])
ocr_result = reader.readtext(binary)
detected_text = ocr_result[0][1] if ocr_result else "N/A"

# --- DISPLAY ---
cv2.rectangle(image_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(image_resized, f"{detected_text} ({conf:.2f})", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- SAVE RESULTS ---
output_image_path = os.path.join(OUTPUT_FOLDER, "detected_plate.jpg")
cv2.imwrite(output_image_path, image_resized)

with open(os.path.join(OUTPUT_FOLDER, "output.json"), "w") as f:
    json.dump({
        "bbox": [x1, y1, x2, y2],
        "confidence": round(conf, 3),
        "text": detected_text
    }, f, indent=2)

with open(os.path.join(OUTPUT_FOLDER, "output.csv"), "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["X1", "Y1", "X2", "Y2", "Confidence", "Detected Text"])
    writer.writerow([x1, y1, x2, y2, round(conf, 3), detected_text])

# --- OUTPUT ---
print(f"Detected Text: {detected_text}")
print(f"Execution Time: {end - start:.2f}s")
