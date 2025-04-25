import cv2
import easyocr
import json
import csv
import os
import time
from ultralytics import YOLO

# Base output directory
base_output_folder = "emb_output"
os.makedirs(base_output_folder, exist_ok=True)

# Create a unique subfolder for this run (e.g., emb_output/run_01/)
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
input_image_path = "HE_image1.jpg"  # Input image

# Load models
model = YOLO(model_path)
reader = easyocr.Reader(['en'])

# Load image
image = cv2.imread(input_image_path)
results = model(image)[0]

# Data storage
detections = []

for i, result in enumerate(results.boxes):
    class_id = int(result.cls[0])
    confidence = float(result.conf[0])
    x1, y1, x2, y2 = map(int, result.xyxy[0])

    # Crop the chassis plate
    cropped = image[y1:y2, x1:x2]
    crop_filename = os.path.join(output_folder, f"crop_{i + 1}.jpg")
    cv2.imwrite(crop_filename, cropped)

    # OCR
    start = time.time()
    ocr_result = reader.readtext(cropped)
    end = time.time()

    text = ocr_result[0][1] if ocr_result else "N/A"
    print(f"[{i + 1}] Text: {text}, Confidence: {confidence:.2f}, Time: {end - start:.2f}s")

    # Draw results
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Save detection
    detections.append({
        "id": i + 1,
        "bounding_box": [x1, y1, x2, y2],
        "confidence": round(confidence, 3),
        "text": text,
        "ocr_time": round(end - start, 2),
        "image_crop": crop_filename
    })

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

# Save and show final annotated image
output_img_path = os.path.join(output_folder, "final_output.jpg")
cv2.imwrite(output_img_path, image)
cv2.imshow("Final Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
