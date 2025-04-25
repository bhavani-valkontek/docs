import os
import cv2
import time
import json
import csv
import torch
import easyocr
import numpy as np
from ultralytics import YOLO

# Paths
image_path = 'HE_image1.jpg'
output_folder = 'emb_output'
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model
model = YOLO('best_news.pt')  # Replace with your trained model

# Load image
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (640, 640))

# Inference
start_time = time.time()
results = model(img_resized)[0]
end_time = time.time()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Store results
detections = []
output_img = img_resized.copy()

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding (better than Otsu for variable lighting)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)

    # Morphology (gentle close to connect characters without losing them)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Optional: Invert if needed
    return cv2.bitwise_not(morph)

# OCR each detected box
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    cropped = img_resized[y1:y2, x1:x2]
    preprocessed = preprocess_for_ocr(cropped)

    ocr_result = reader.readtext(preprocessed, detail=0)
    recognized_text = ' '.join(ocr_result).strip()

    detections.append({
        'box': [x1, y1, x2, y2],
        'confidence': round(conf, 3),
        'text': recognized_text
    })

    # Draw box and text
    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(output_img, recognized_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Save output image
output_image_path = os.path.join(output_folder, 'output.jpg')
cv2.imwrite(output_image_path, output_img)

# Save CSV
csv_path = os.path.join(output_folder, 'detections.csv')
with open(csv_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['box', 'confidence', 'text'])
    writer.writeheader()
    for det in detections:
        writer.writerow(det)

# Save JSON
json_path = os.path.join(output_folder, 'detections.json')
with open(json_path, 'w') as file:
    json.dump(detections, file, indent=2)

# Print summary
print("✅ Inference completed.")
print(f"⏱️ Execution time: {round(end_time - start_time, 2)}s")
print(f"📝 Detections saved to {csv_path} and {json_path}")
for d in detections:
    print(f"🔍 Text: {d['text']} | Confidence: {d['confidence']}")
