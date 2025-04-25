import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import easyocr
from ultralytics import YOLO
import csv
import datetime

# Global Variables
image_path = None
model = None
original_image = None
processed_image = None
display_image = None
reader = easyocr.Reader(['en'], gpu=False)
zoom_level = 1.0
rotation_angle = 0

# ---------------- GUI Functions ---------------- #

def load_model():
    global model
    model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("Model files", "*.pt")])
    if model_path:
        model = YOLO(model_path)
        model_label.config(text=f"Model: {os.path.basename(model_path)}")

def load_image():
    global image_path, original_image, zoom_level, rotation_angle
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        original_image = cv2.imread(image_path)
        zoom_level = 1.0
        rotation_angle = 0
        update_display_image()

def update_display_image():
    global display_image
    if original_image is None:
        return
    img = original_image.copy()

    if rotation_angle != 0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    if zoom_level != 1.0:
        img = cv2.resize(img, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)

    display_image = img
    show_image(img)

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((500, 500))
    imgtk = ImageTk.PhotoImage(img_pil)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

def run_detection():
    global processed_image, display_image
    if model is None or display_image is None:
        messagebox.showwarning("Warning", "Please load a model and an image first.")
        return

    conf_threshold = confidence_slider.get()
    results = model(display_image, conf=conf_threshold)
    result = results[0]
    boxes = result.boxes
    image_copy = display_image.copy()

    detected_texts = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image_copy[y1:y2, x1:x2]
        ocr_result = reader.readtext(cropped)
        if ocr_result:
            text = ocr_result[0][1]
            detected_texts.append((text, x1, y1, x2, y2))
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    processed_image = image_copy
    show_image(processed_image)

    text_output.delete("1.0", tk.END)
    for text, _, _, _, _ in detected_texts:
        text_output.insert(tk.END, text + "\n")

def save_output():
    global processed_image
    if processed_image is None:
        messagebox.showwarning("Warning", "No processed image to save.")
        return

    folder = filedialog.askdirectory(title="Select Folder to Save Outputs")
    if not folder:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = os.path.join(folder, f"detection_{timestamp}.jpg")
    cv2.imwrite(image_filename, processed_image)

    text = text_output.get("1.0", tk.END).strip()
    csv_filename = os.path.join(folder, f"text_{timestamp}.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for line in text.split("\n"):
            writer.writerow([line])

    messagebox.showinfo("Saved", f"Saved image and text to:\n{folder}")

def zoom_in():
    global zoom_level
    zoom_level *= 1.2
    update_display_image()

def zoom_out():
    global zoom_level
    zoom_level /= 1.2
    update_display_image()

def rotate_left():
    global rotation_angle
    rotation_angle -= 90
    update_display_image()

def rotate_right():
    global rotation_angle
    rotation_angle += 90
    update_display_image()

# ---------------- GUI Layout ---------------- #

root = tk.Tk()
root.title("YOLO + OCR Detection GUI")

# ----- TOP CONTROL PANEL ----- #
top_controls_frame = tk.LabelFrame(root, text="Controls", padx=10, pady=5)
top_controls_frame.pack(fill="x", padx=10, pady=10)

tk.Button(top_controls_frame, text="Load YOLO Model", command=load_model).grid(row=0, column=0, padx=5)
model_label = tk.Label(top_controls_frame, text="Model: None")
model_label.grid(row=0, column=1, padx=5)

tk.Button(top_controls_frame, text="Load Image", command=load_image).grid(row=0, column=2, padx=5)
tk.Label(top_controls_frame, text="Confidence").grid(row=0, column=3, padx=5)

confidence_slider = tk.Scale(top_controls_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
confidence_slider.set(0.5)
confidence_slider.grid(row=0, column=4, padx=5)

tk.Button(top_controls_frame, text="Zoom In", command=zoom_in).grid(row=0, column=5, padx=5)
tk.Button(top_controls_frame, text="Zoom Out", command=zoom_out).grid(row=0, column=6, padx=5)
tk.Button(top_controls_frame, text="⟲ Rotate Left", command=rotate_left).grid(row=0, column=7, padx=5)
tk.Button(top_controls_frame, text="⟳ Rotate Right", command=rotate_right).grid(row=0, column=8, padx=5)
tk.Button(top_controls_frame, text="Run Detection", command=run_detection).grid(row=0, column=9, padx=5)

# ----- IMAGE DISPLAY ----- #
image_label = tk.Label(root)
image_label.pack(padx=10, pady=5)

# ----- TEXT OUTPUT ----- #
text_frame = tk.LabelFrame(root, text="Detected Text (Editable)")
text_frame.pack(padx=10, pady=10, fill="both", expand=True)

text_output = tk.Text(text_frame, height=10, wrap="word")
text_output.pack(fill="both", expand=True)

# ----- SAVE BUTTON ----- #
save_btn = tk.Button(root, text="💾 Save Output (Image + Text)", command=save_output)
save_btn.pack(pady=10)

root.mainloop()
