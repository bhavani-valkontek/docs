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

# Constants
DEFAULT_MODEL_PATH = "best_b.pt"
DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
TOTAL_CLASSES = 80  # Change this if using a custom model

def get_next_image_number(output_dir, prefix, suffix):
    existing = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(suffix)]
    numbers = [int(f.replace(prefix, "").replace(suffix, "")) for f in existing if f.replace(prefix, "").replace(suffix, "").isdigit()]
    return max(numbers, default=0) + 1

def get_color_for_label(label):
    random.seed(hash(label) % 1000)
    return tuple(random.randint(0, 255) for _ in range(3))

def capture_image():
    cap = cv2.VideoCapture(0)

    # Set the webcam resolution to 1280x960
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None

    print("[INFO] Press 'c' to capture an image or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Resize only for display
        display_frame = cv2.resize(frame, (1024, 1024))
        cv2.imshow("Live Feed (1280x960)", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return frame  # Return original frame for processing
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def process_image(original_img, yolo_model, confidence_threshold, output_dir, csv_path, show_images, start_time):
    try:
        if original_img is None:
            print("[ERROR] No image provided.")
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_resized = cv2.resize(original_img, (1024, 1024))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        print("[INFO] Running YOLO model inference...")
        results = yolo_model.predict(img_rgb, verbose=False)

        scale_x = original_img.shape[1] /1024
        scale_y = original_img.shape[0] /1024

        detection_count = 0
        used_label_positions = []
        all_detected_labels = []
        detection_json_data = []

        image_all_detections = original_img.copy()
        image_filtered_detections = original_img.copy()

        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = result.names[int(cls)]
                conf_val = float(conf)
                conf_text = f"{label} ({conf_val:.2f})"
                color = get_color_for_label(label)

                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                def draw_on_image(img):
                    cv2.rectangle(img, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                    (tw, th), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = y1_orig - 10 if y1_orig - 10 > th else y2_orig + 10
                    while any(abs(label_y - used) < th + 5 for used in used_label_positions):
                        label_y += th + 5
                        if label_y + th > img.shape[0]:
                            label_y = y1_orig + th + 10
                            break
                    used_label_positions.append(label_y)
                    cv2.rectangle(img, (x1_orig, label_y - th), (x1_orig + tw + 4, label_y + 4), color, -1)
                    cv2.putText(img, conf_text, (x1_orig + 2, label_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                draw_on_image(image_all_detections)

                if conf_val > confidence_threshold:
                    detection_count += 1
                    all_detected_labels.append(label)
                    draw_on_image(image_filtered_detections)

                    detection_json_data.append({
                        "label": label,
                        "confidence": round(conf_val, 2),
                        "bounding_box": [x1_orig, y1_orig, x2_orig, y2_orig]
                    })

        if detection_count == 0:
            print("[INFO] No detections above confidence threshold.")
        else:
            print(f"[INFO] Total valid detections: {detection_count}")

        os.makedirs(output_dir, exist_ok=True)
        img_number = get_next_image_number(output_dir, "image_", "_all.jpg")
        all_output_path = os.path.join(output_dir, f"image_{img_number}_all.jpg")
        conf_output_path = os.path.join(output_dir, f"image_{img_number}_conf.jpg")
        json_output_path = os.path.join(output_dir, f"image_{img_number}.json")

        cv2.imwrite(all_output_path, image_all_detections)
        cv2.imwrite(conf_output_path, image_filtered_detections)

        label_counts = Counter(all_detected_labels)
        formatted_labels = []
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            formatted_labels.append(f"{label}({count})" if count > 1 else label)

        percentage_detected = (len(label_counts.keys()) / TOTAL_CLASSES) * 100

        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["timestamp", "detected_classes", "total_detected_classes", "percentage_detected"])
            writer.writerow([
                timestamp,
                ", ".join(formatted_labels),
                len(label_counts.keys()),
                f"{percentage_detected:.2f}%"
            ])

        # Save JSON
        with open(json_output_path, 'w') as json_file:
            json.dump({
                "timestamp": timestamp,
                "detections": detection_json_data
            }, json_file, indent=4)

        print(f"[INFO] Saved all detections image to: {all_output_path}")
        print(f"[INFO] Saved filtered detections image to: {conf_output_path}")
        print(f"[INFO] Saved JSON to: {json_output_path}")
        print(f"[INFO] CSV appended to: {csv_path}")

        total_time = time.time() - start_time
        print(f"[INFO] Total processing time: {total_time:.2f} seconds")

        if show_images:
            try:
                cv2.imshow("All Detections", image_all_detections)
                cv2.imshow("Filtered Detections", image_filtered_detections)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                print("[WARNING] Cannot display image windows in headless environment.")

    except Exception as e:
        print(f"[ERROR] While processing image: {e}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Webcam or Image Inference Tool")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to YOLOv8 model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--image", type=str, help="Path to an image (skip camera)")
    parser.add_argument("--output", type=str, default="output", help="Directory to save results")
    parser.add_argument("--no-show", action="store_true", help="Disable image preview (headless mode)")
    args = parser.parse_args()

    try:
        if args.image:
            if not os.path.exists(args.image):
                print(f"[ERROR] Provided image not found: {args.image}")
                return
            original_img = cv2.imread(args.image)
            print(f"[INFO] Using provided image: {args.image}")
        else:
            original_img = capture_image()

        if original_img is None:
            print("[ERROR] No image available for processing.")
            return

        print("[INFO] Loading YOLOv8 model...")
        yolo_model = YOLO(args.model).to(DEVICE)
        print("[INFO] YOLO model loaded.")

        start_time = time.time()
        csv_path = os.path.join(args.output, "detections.csv")
        process_image(
            original_img,
            yolo_model,
            args.conf,
            args.output,
            csv_path,
            show_images=not args.no_show,
            start_time=start_time
        )

    except Exception as e:
        print(f"[FATAL ERROR] {e}")



if __name__ == "__main__":
    main()