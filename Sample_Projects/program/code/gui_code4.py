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
display_image = None
reader = easyocr.Reader(['en'], gpu=False)
zoom_level = 1.0
rotation_angle = 0
pan_start = None

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
        image_label_path.config(text=f"Image: {os.path.basename(image_path)}")
        update_display_image()

def update_display_image():
    global display_image
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

    display_image = img
    show_image(img)

def show_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((600, 600))
    imgtk = ImageTk.PhotoImage(img_pil)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

def run_detection():
    global processed_image, display_image
    if model is None or display_image is None:
        messagebox.showwarning("Warning", "Please load a model and an image first.")
        return

    try:
        conf_threshold = float(conf_entry.get())
    except ValueError:
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

    # Update text output
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

    # Save text
    text = text_output.get("1.0", tk.END).strip()
    csv_filename = os.path.join(folder, f"text_{timestamp}.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for line in text.split("\n"):
            writer.writerow([line])

    messagebox.showinfo("Saved", f"Saved image and text to:\n{folder}")

def update_zoom(val):
    global zoom_level
    try:
        zoom_level = float(zoom_entry.get())
    except ValueError:
        zoom_level = float(val)
    update_display_image()

def on_mouse_down(event):
    global pan_start
    pan_start = (event.x, event.y)

def on_mouse_move(event):
    global pan_start, image_label
    if pan_start:
        dx = event.x - pan_start[0]
        dy = event.y - pan_start[1]
        image_label.place(x=image_label.winfo_x() + dx, y=image_label.winfo_y() + dy)
        pan_start = (event.x, event.y)

def on_mouse_up(event):
    global pan_start
    pan_start = None

# ---------------- GUI Layout ---------------- #

root = tk.Tk()
root.title("Professional YOLO + OCR GUI")
root.geometry("900x800")

# --- Top Frame --- #
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

model_btn = tk.Button(top_frame, text="Load Model", command=load_model)
model_btn.grid(row=0, column=0, padx=5)

model_label = tk.Label(top_frame, text="Model: None", anchor='w', width=25)
model_label.grid(row=0, column=1, padx=5)

image_btn = tk.Button(top_frame, text="Load Image", command=load_image)
image_btn.grid(row=0, column=2, padx=5)

image_label_path = tk.Label(top_frame, text="Image: None", anchor='w', width=25)
image_label_path.grid(row=0, column=3, padx=5)

confidence_slider = tk.Scale(top_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, label="Confidence")
confidence_slider.set(0.5)
confidence_slider.grid(row=1, column=0, padx=5)

conf_entry = tk.Entry(top_frame, width=5)
conf_entry.insert(0, "0.5")
conf_entry.grid(row=1, column=1, padx=5)

zoom_slider = tk.Scale(top_frame, from_=0.5, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Zoom", command=update_zoom)
zoom_slider.set(1.0)
zoom_slider.grid(row=1, column=2, padx=5)

zoom_entry = tk.Entry(top_frame, width=5)
zoom_entry.insert(0, "1.0")
zoom_entry.grid(row=1, column=3, padx=5)

run_btn = tk.Button(top_frame, text="▶ Run Detection", command=run_detection)
run_btn.grid(row=1, column=4, padx=10)

# --- Center Frame (Image) --- #
image_frame = tk.Frame(root, width=600, height=600)
image_frame.pack(pady=10)

image_label = tk.Label(image_frame)
image_label.pack()
image_label.bind("<ButtonPress-1>", on_mouse_down)
image_label.bind("<B1-Motion>", on_mouse_move)
image_label.bind("<ButtonRelease-1>", on_mouse_up)

# --- Bottom Frame (Text + Save) --- #
text_frame = tk.Frame(root)
text_frame.pack(padx=10, pady=10, fill="both", expand=True)

text_output = tk.Text(text_frame, height=10, wrap="word")
text_output.pack(side=tk.LEFT, fill="both", expand=True, padx=5)

btn_frame = tk.Frame(text_frame)
btn_frame.pack(side=tk.RIGHT, padx=5)

save_btn = tk.Button(btn_frame, text="💾 Save Output", command=save_output)
save_btn.pack(pady=5)

root.mainloop()
