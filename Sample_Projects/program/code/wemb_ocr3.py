import cv2
import easyocr
import json
import csv
import os
import time
import numpy as np
from ultralytics import YOLO

# Base output directory
base_output_folder = "emb_output"
os.makedirs(base_output_folder, exist_ok=True)


def create_unique_run_folder(base_folder):
    run_id = 1
    while True:
        run_folder = os.path.join(base_folder, f"run_{run_id:02d}")
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
            return run_folder
        run_id += 1


# Create new run folder
output_folder = create_unique_run_folder(base_output_folder)

# Paths and setup
model_path = "best_news.pt"  # Your trained YOLOv8 model
input_image_path = "img30.jpg"  # Input image

# Initialize models
model = YOLO(model_path)
reader = easyocr.Reader(['en'])  # Start with default settings


def simple_preprocess(image):
    """Simpler preprocessing that works for most cases"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    return gray


# Load image
image = cv2.imread(input_image_path)
if image is None:
    raise ValueError(f"Could not load image from {input_image_path}")

# Run detection
results = model(image)[0]

# Data storage
detections = []

for i, result in enumerate(results.boxes):
    class_id = int(result.cls[0])
    confidence = float(result.conf[0])
    x1, y1, x2, y2 = map(int, result.xyxy[0])

    # Small expansion of bounding box (5%)
    h, w = image.shape[:2]
    expand = int(0.05 * (x2 - x1))
    x1 = max(0, x1 - expand)
    y1 = max(0, y1 - expand)
    x2 = min(w, x2 + expand)
    y2 = min(h, y2 + expand)

    # Crop the license plate
    cropped = image[y1:y2, x1:x2]

    # Save original crop
    crop_filename = os.path.join(output_folder, f"crop_{i + 1}.jpg")
    cv2.imwrite(crop_filename, cropped)

    # Simple preprocessing
    processed = simple_preprocess(cropped)
    processed_filename = os.path.join(output_folder, f"processed_{i + 1}.jpg")
    cv2.imwrite(processed_filename, processed)

    # OCR with basic settings
    start = time.time()
    try:
        ocr_result = reader.readtext(
            processed,
            detail=0,
            paragraph=True,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        text = " ".join(ocr_result) if ocr_result else "N/A"
    except Exception as e:
        print(f"OCR failed: {e}")
        text = "N/A"
    end = time.time()

    # Simple text cleaning
    clean_text = "".join([c for c in text.upper() if c.isalnum()])

    print(f"[{i + 1}] Text: {clean_text}, Confidence: {confidence:.2f}, Time: {end - start:.2f}s")

    # Draw results
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, clean_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save detection
    detections.append({
        "id": i + 1,
        "bounding_box": [x1, y1, x2, y2],
        "confidence": round(confidence, 3),
        "text": clean_text,
        "ocr_time": round(end - start, 2),
        "image_crop": crop_filename,
        "processed_image": processed_filename
    })

# Save outputs only if detections were found
if detections:
    # Save to CSV
    csv_path = os.path.join(output_folder, "detections.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=detections[0].keys())
        writer.writeheader()
        writer.writerows(detections)

    # Save to JSON
    json_path = os.path.join(output_folder, "detections.json")
    with open(json_path, "w") as jsonfile:
        json.dump(detections, jsonfile, indent=4)
else:
    print("No license plates detected!")

# Save and show final annotated image
output_img_path = os.path.join(output_folder, "final_output.jpg")
cv2.imwrite(output_img_path, image)
cv2.imshow("Final Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()