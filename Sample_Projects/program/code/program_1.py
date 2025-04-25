import os
import cv2
import torch
import random
import hashlib
from ultralytics import YOLO

# ====================== Constants ======================
IMAGE_PATH = "test_image.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.25

# ====================== Utility ======================
def get_color_for_label(label):
    random.seed(int(hashlib.sha256(label.encode()).hexdigest(), 16))
    return tuple(random.randint(0, 255) for _ in range(3))

# ====================== YOLOv8 OCR ======================
def process_image(original_img, yolo_model):
    try:
        if original_img is None:
            print("[ERROR] No image provided.")
            return

        img_resized = cv2.resize(original_img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        print("[INFO] Running YOLO model inference...")
        results = yolo_model(img_rgb)

        scale_x = original_img.shape[1] / 640
        scale_y = original_img.shape[0] / 640

        detection_count = 0
        used_label_positions = []

        # Clone originals
        img_with_boxes_all = original_img.copy()
        img_with_boxes_threshold = original_img.copy()

        for result in results:
            print(f"[INFO] Processing {len(result.boxes)} detected boxes...")
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                label = result.names[int(cls)]
                conf_text = f"{label} ({conf:.2f})"
                color = get_color_for_label(label)

                # Rescale to original size
                x1_orig = int(x1 * scale_x)
                y1_orig = int(y1 * scale_y)
                x2_orig = int(x2 * scale_x)
                y2_orig = int(y2 * scale_y)

                # Common drawing code
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

                    label_bg_top_left = (x1_orig, label_y - th)
                    label_bg_bottom_right = (x1_orig + tw + 4, label_y + 4)
                    cv2.rectangle(img, label_bg_top_left, label_bg_bottom_right, color, -1)
                    cv2.putText(img, conf_text, (x1_orig + 2, label_y + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Draw on all-detection image
                draw_on_image(img_with_boxes_all)

                if float(conf) > CONFIDENCE_THRESHOLD:
                    detection_count += 1
                    draw_on_image(img_with_boxes_threshold)
                    print(f"[DETECTION] {conf_text} at ({x1_orig}, {y1_orig}, {x2_orig}, {y2_orig})")

        if detection_count == 0:
            print("[INFO] No detections above confidence threshold.")
        else:
            print(f"[INFO] Total valid detections: {detection_count}")

        os.makedirs("output", exist_ok=True)

        # Save both images
        all_output_path = "output/image_all_boxes.jpg"
        conf_output_path = "output/image_confidence_filtered.jpg"

        cv2.imwrite(all_output_path, img_with_boxes_all)
        cv2.imwrite(conf_output_path, img_with_boxes_threshold)

        print(f"[INFO] Saved all boxes image to: {all_output_path}")
        print(f"[INFO] Saved confidence filtered image to: {conf_output_path}")

        # Show images
        cv2.imshow("All Detections", img_with_boxes_all)
        cv2.imshow("Filtered Detections", img_with_boxes_threshold)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] While processing image: {e}")

# ====================== Camera ======================
def capture_image():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_GAIN, 7)

    print("[INFO] Camera initialized. Press 'c' to capture image.")
    original_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break
        cv2.imshow("Live YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            original_frame = frame.copy()
            cv2.imwrite(IMAGE_PATH, cv2.resize(frame, (640, 640)))  # Optional
            print(f"[INFO] Image captured and saved to: {IMAGE_PATH}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return original_frame

# ====================== Main ======================
def main():
    try:
        original_img = capture_image()
        print("[INFO] Loading YOLOv8 model...")
        yolo_model = YOLO("sub/best.pt")  # Replace with your model path
        print("[INFO] YOLO model loaded successfully.")
        process_image(original_img, yolo_model)
    except Exception as e:
        print(f"[FATAL ERROR] {e}")

if __name__ == "__main__":
    main()