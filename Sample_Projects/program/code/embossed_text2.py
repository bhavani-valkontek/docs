import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import csv
import json

def enhance_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Bilateral Filter (preserves edges)
    filtered = cv2.bilateralFilter(contrast, 9, 80, 80)

    # Morphological gradient to highlight edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)

    # Normalize and merge to 3 channels
    norm = cv2.normalize(morph, None, 0, 255, cv2.NORM_MINMAX)
    processed = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

    # Resize to YOLO expected size
    resized = cv2.resize(processed, (768, 768))
    return resized

def detect_chassis_number(image_path, confidence_threshold=0.25):
    model = YOLO("best_news.pt")  # your trained YOLO model
    output_dir = "emb_output"
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    enhanced_image = enhance_image(image)

    # Inference
    start = time.time()
    results = model.predict([enhanced_image], conf=confidence_threshold)
    elapsed = time.time() - start

    detections = []
    result = results[0]

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.rectangle(enhanced_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(enhanced_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detections.append({
            "label": model.names[cls],
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

        print(f"[DETECTED] {model.names[cls]} - Confidence: {conf:.2f}, Box: {x1},{y1},{x2},{y2}")

    # Save outputs
    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base}_output.jpg"), enhanced_image)

    # CSV
    with open(os.path.join(output_dir, f"{base}_detections.csv"), "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["label", "confidence", "bbox"])
        writer.writeheader()
        writer.writerows(detections)

    # JSON
    with open(os.path.join(output_dir, f"{base}_detections.json"), "w") as jf:
        json.dump(detections, jf, indent=4)

    print(f"\n✅ Output saved to {output_dir}")
    print(f"⚙️ Inference Time: {elapsed:.2f} seconds")
    print(f"🎯 Confidence Threshold: {confidence_threshold}")

    return result

# Run detection
if __name__ == "__main__":
    image_path = "chassis.jpg"  # Replace with your image
    detect_chassis_number(image_path)
