import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
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
resized_display_image = None
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
        image_label.config(text=f"Image: {os.path.basename(image_path)}")
        zoom_level = 1.0
        rotation_angle = 0
        update_display_image()

def update_display_image():
    global resized_display_image
    if original_image is None:
        return
    img = original_image.copy()

    # Apply rotation
    if rotation_angle != 0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1)
        img = cv2.warpAffine(img, M, (w, h))

    # Apply zoom
    if zoom_level != 1.0:
        img = cv2.resize(img, None, fx=zoom_level, fy=zoom_level, interpolation=cv2.INTER_LINEAR)

    resized_display_image = img
    show_image(img)

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(img_pil)
    canvas.imgtk = imgtk
    canvas.config(width=imgtk.width(), height=imgtk.height())
    canvas.create_image(0, 0, anchor="nw", image=imgtk)

def run_detection():
    global processed_image, resized_display_image
    if model is None or resized_display_image is None:
        messagebox.showwarning("Warning", "Please load a model and an image first.")
        return

    conf_threshold = confidence_slider.get()
    results = model(resized_display_image, conf=conf_threshold)
    result = results[0]
    boxes = result.boxes
    image_copy = resized_display_image.copy()

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
    text_filename = os.path.join(folder, f"text_{timestamp}.txt")
    with open(text_filename, "w") as f:
        f.write(text)

    messagebox.showinfo("Saved", f"Saved image and text to:\n{folder}")

def save_to_csv():
    text = text_output.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "No text to save.")
        return

    file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not file:
        return

    with open(file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for line in text.split("\n"):
            writer.writerow([line])

    messagebox.showinfo("Saved", f"Saved CSV to:\n{file}")

def update_zoom(val):
    global zoom_level
    try:
        zoom_level = float(val)
        update_display_image()
    except ValueError:
        pass

def update_confidence(val):
    try:
        confidence_slider.set(float(val))
    except ValueError:
        pass

# ---------------- GUI Layout ---------------- #

root = tk.Tk()
root.title("YOLO + OCR Detection GUI")

# Top Frame
top_frame = tk.Frame(root)
top_frame.pack(side="top", fill="x", padx=10, pady=5)

model_btn = tk.Button(top_frame, text="Load YOLO Model", command=load_model)
model_btn.pack(side="left")
model_label = tk.Label(top_frame, text="Model: None", width=25, anchor='w')
model_label.pack(side="left", padx=(5, 15))

image_btn = tk.Button(top_frame, text="Load Image", command=load_image)
image_btn.pack(side="left")
image_label = tk.Label(top_frame, text="Image: None", width=25, anchor='w')
image_label.pack(side="left", padx=(5, 15))

run_btn = tk.Button(top_frame, text="Run Detection", command=run_detection)
run_btn.pack(side="left", padx=5)

zoom_label = tk.Label(top_frame, text="Zoom:")
zoom_label.pack(side="left")
zoom_slider = tk.Scale(top_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, command=update_zoom)
zoom_slider.set(1.0)
zoom_slider.pack(side="left")
zoom_entry = tk.Entry(top_frame, width=5)
zoom_entry.insert(0, "1.0")
zoom_entry.bind("<Return>", lambda event: update_zoom(zoom_entry.get()))
zoom_entry.pack(side="left", padx=5)

conf_label = tk.Label(top_frame, text="Confidence:")
conf_label.pack(side="left")
confidence_slider = tk.Scale(top_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL)
confidence_slider.set(0.5)
confidence_slider.pack(side="left")
conf_entry = tk.Entry(top_frame, width=5)
conf_entry.insert(0, "0.5")
conf_entry.bind("<Return>", lambda event: update_confidence(conf_entry.get()))
conf_entry.pack(side="left", padx=5)

# Canvas for image display
canvas = tk.Canvas(root)
canvas.pack(pady=10)

# Bottom Text Frame
bottom_frame = tk.Frame(root)
bottom_frame.pack(fill="both", expand=True, padx=10, pady=10)

text_output = tk.Text(bottom_frame, height=10, wrap="word")
text_output.pack(side="left", fill="both", expand=True)

btn_frame = tk.Frame(bottom_frame)
btn_frame.pack(side="left", padx=10)

save_btn = tk.Button(btn_frame, text="💾 Save Image + Text", command=save_output)
save_btn.pack(pady=5)

csv_btn = tk.Button(btn_frame, text="📁 Save to CSV", command=save_to_csv)
csv_btn.pack(pady=5)

root.mainloop()
