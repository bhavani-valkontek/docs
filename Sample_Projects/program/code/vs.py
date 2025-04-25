import streamlit as st
import os
import json
from PIL import Image, ImageDraw, ImageFont


# Helper to draw boxes
def draw_boxes(image_path, detections):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for box in detections.get("boxes", []):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        label = box["label"]
        confidence = box["confidence"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{label} ({confidence:.2f})", fill="yellow")

    return image


# UI
st.title("📦 YOLO Detection Visualizer")

# Select directory
output_dir = st.text_input("Enter the output folder path:", value="outputs")

if os.path.isdir(output_dir):
    images = [f for f in os.listdir(output_dir) if f.endswith(".jpg") or f.endswith(".png")]

    if images:
        selected_image = st.selectbox("Choose an image to visualize:", images)
        image_path = os.path.join(output_dir, selected_image)
        json_path = os.path.splitext(image_path)[0] + ".json"

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                detection_data = json.load(f)

            st.subheader("📸 Original Image with Detections")
            vis_image = draw_boxes(image_path, detection_data)
            st.image(vis_image, caption="Detections", use_column_width=True)

            # Show raw JSON
            with st.expander("🔍 Raw Detection Data"):
                st.json(detection_data)

            # Optional: bar chart of confidence
            confidences = [box["confidence"] for box in detection_data.get("boxes", [])]
            labels = [box["label"] for box in detection_data.get("boxes", [])]

            if confidences:
                st.subheader("📊 Confidence Scores")
                st.bar_chart(data={"Confidence": confidences}, use_container_width=True)
    else:
        st.warning("No image files found in that directory.")
else:
    st.warning("Please enter a valid directory path.")
