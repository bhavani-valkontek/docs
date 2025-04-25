import cv2
import numpy as np
from ultralytics import YOLO
import os

def embossed_text_detection_pipeline(image_path, confidence_threshold=0.25, save_output=True):
    # Load the model
    model = YOLO("best_news.pt")  # Replace with your actual trained model path

    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Could not read image from: {image_path}")

    # Run prediction (wrap the image in a list)
    results = model.predict([image], conf=confidence_threshold)

    # Get the result for the first image
    result = results[0]

    # Draw boxes and labels on the image
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"[DETECTED] Class: {label}, Box: ({x1}, {y1}, {x2}, {y2})")

    # Save output image
    if save_output:
        out_path = os.path.splitext(image_path)[0] + "_output.jpg"
        cv2.imwrite(out_path, image)
        print(f"\n✅ Output image saved to: {out_path}")
    else:
        print("Skipping image save (set save_output=True to save).")

    return result

# === Main call ===
if __name__ == "__main__":
    image_path = "chassis.jpg"  # Replace with your image path
    embossed_text_detection_pipeline(image_path)
