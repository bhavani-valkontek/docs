import cv2
import os
import csv
import time
import json
import argparse
import random
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import pytesseract

# Optional: Tesseract path for Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def recognize_text(image):
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(image, config=config).strip()

def detect_and_recognize(model_path, image_path, output_dir):
    # Load YOLO model
    model = YOLO(model_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run detection
    results = model(image)[0]
    recognized_texts = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if conf > 0.5:
            roi = image[y1:y2, x1:x2]
            processed = preprocess_image(roi)
            text = recognize_text(processed)

            recognized_texts.append({
                "text": text,
                "confidence": conf,
                "box": [x1, y1, x2, y2]
            })

            # Draw on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save output image
    output_img_path = os.path.join(output_dir, f"output_{timestamp}.png")
    cv2.imwrite(output_img_path, image)

    # Save JSON
    json_path = os.path.join(output_dir, f"result_{timestamp}.json")
    with open(json_path, "w") as jf:
        json.dump(recognized_texts, jf, indent=4)

    # Save CSV
    csv_path = os.path.join(output_dir, f"result_{timestamp}.csv")
    with open(csv_path, mode="w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["Text", "Confidence", "X1", "Y1", "X2", "Y2"])
        for item in recognized_texts:
            writer.writerow([item["text"], item["confidence"]] + item["box"])

    # Simple analysis with Counter and random dummy data
    chars = ''.join([item["text"] for item in recognized_texts])
    counter = Counter(chars)
    print("Character Frequency:", dict(counter))
    print("Random ID for log:", random.randint(1000, 9999))

    return recognized_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chassis Number Detection & Recognition App")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model (best.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to save results")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    start = time.time()
    results = detect_and_recognize(args.model, args.image, args.output)
    end = time.time()

    print("\n--- Detection Complete ---")
    print(f"Detected {len(results)} regions")
    for r in results:
        print(f"Text: {r['text']} | Confidence: {r['confidence']:.2f}")
    print(f"Execution Time: {end - start:.2f} seconds")
