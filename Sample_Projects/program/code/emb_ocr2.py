import csv
import json
import os
import re
import time
from ultralytics import YOLO

import cv2
import easyocr
import numpy as np

# Base output directory
base_output_folder = "emb_outpt"
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
input_image_path = "HE_image.jpg"  # Input image

# Load models
model = YOLO(model_path)
reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU if available


# Custom preprocessing function for better OCR results
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Remove noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return processed


# License plate text validation and formatting
def validate_plate_text(text):
    # Remove spaces and special characters
    clean_text = re.sub(r'[^a-zA-Z0-9]', '', text.upper())

    # Common license plate patterns (customize for your region)
    patterns = [
        r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$',  # DL-01-AB-1234 format
        r'^[A-Z]{3}[0-9]{4}$',  # ABC-1234 format
        r'^[0-9]{4}[A-Z]{3}$',  # 1234-ABC format
        r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'  # Variants
    ]

    # Check if any pattern matches
    for pattern in patterns:
        if re.match(pattern, clean_text):
            return clean_text

    # If no pattern matches, return the cleaned text anyway
    return clean_text if len(clean_text) >= 4 else "N/A"


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

    # Expand bounding box slightly (10% expansion)
    h, w = image.shape[:2]
    x1 = max(0, x1 - int(0.1 * (x2 - x1)))
    y1 = max(0, y1 - int(0.1 * (y2 - y1)))
    x2 = min(w, x2 + int(0.1 * (x2 - x1)))
    y2 = min(h, y2 + int(0.1 * (y2 - y1)))

    # Crop the license plate
    cropped = image[y1:y2, x1:x2]

    # Preprocess for better OCR
    processed = preprocess_image(cropped)

    crop_filename = os.path.join(output_folder, f"crop_{i + 1}.jpg")
    cv2.imwrite(crop_filename, cropped)

    # Save processed image for debugging
    processed_filename = os.path.join(output_folder, f"processed_{i + 1}.jpg")
    cv2.imwrite(processed_filename, processed)

    # OCR with detailed configuration
    start = time.time()
    ocr_result = reader.readtext(
        processed,
        decoder='beamsearch',  # Better than greedy for plates
        beamWidth=5,
        batch_size=1,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',  # Only allow alphanumeric
        detail=0,
        paragraph=True
    )
    end = time.time()

    # Process OCR results
    text = " ".join(ocr_result) if ocr_result else "N/A"
    validated_text = validate_plate_text(text)

    print(
        f"[{i + 1}] Raw Text: {text}, Validated: {validated_text}, Confidence: {confidence:.2f}, Time: {end - start:.2f}s")

    # Draw results
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, validated_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save detection
    detections.append({
        "id": i + 1,
        "bounding_box": [x1, y1, x2, y2],
        "confidence": round(confidence, 3),
        "raw_text": text,
        "validated_text": validated_text,
        "ocr_time": round(end - start, 2),
        "image_crop": crop_filename,
        "processed_image": processed_filename
    })

# Save to CSV
csv_path = os.path.join(output_folder, "detections.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    fieldnames = detections[0].keys() if detections else []
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    if detections:
        writer.writerows(detections)

# Save to JSON
json_path = os.path.join(output_folder, "detections.json")
with open(json_path, "w") as jsonfile:
    json.dump(detections, jsonfile, indent=4)

# Save and show final annotated image
output_img_path = os.path.join(output_folder, "final_output.jpg")
cv2.imwrite(output_img_path, image)
cv2.imshow("Final Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()