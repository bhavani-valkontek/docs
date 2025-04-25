import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk, ImageOps
import cv2
import easyocr
from ultralytics import YOLO
import numpy as np
import csv
import json

# Globals
model = None
ocr_reader = easyocr.Reader(['en'])
image_path = ""
original_image = None
display_image_tk = None
output_text = []
rotation_angle = 0
zoom_factor = 1.0

# Load model
def load_model():
    global model
    model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("YOLO Model", "*.pt")])
    if model_path:
        try:
            model = YOLO(model_path)
            messagebox.showinfo("Model Loaded", f"Model loaded successfully:\n{model_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")

# Load image
def load_image():
    global image_path, original_image, rotation_angle, zoom_factor
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if image_path:
        rotation_angle = 0
        zoom_factor = 1.0
        original_image = Image.open(image_path)
        update_display_image(original_image)

# Update image on panel
def update_display_image(pil_image):
    global display_image_tk
    resized = pil_image.copy()
    width, height = resized.size
    resized = resized.resize((int(width * zoom_factor), int(height * zoom_factor)))
    display_image_tk = ImageTk.PhotoImage(resized)
    panel.config(image=display_image_tk)
    panel.image = display_image_tk

# Zoom controls
def zoom_in():
    global zoom_factor
    zoom_factor *= 1.1
    update_display_image(apply_rotation(original_image))

def zoom_out():
    global zoom_factor
    zoom_factor /= 1.1
    update_display_image(apply_rotation(original_image))

# Rotation controls
def rotate_left():
    global rotation_angle
    rotation_angle = (rotation_angle - 90) % 360
    update_display_image(apply_rotation(original_image))

def rotate_right():
    global rotation_angle
    rotation_angle = (rotation_angle + 90) % 360
    update_display_image(apply_rotation(original_image))

def apply_rotation(img):
    return img.rotate(rotation_angle, expand=True)

# Run detection
def run_detection():
    global output_text
    if model is None:
        messagebox.showwarning("Warning", "Please load a YOLO model first.")
        return
    if not image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return

    conf = threshold_slider.get() / 100.0

    image = cv2.imread(image_path)
    rotated = apply_rotation(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    rotated_cv2 = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2BGR)

    results = model(rotated_cv2, conf=conf)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = rotated_cv2[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        ocr_result = ocr_reader.readtext(cropped)
        for (_, text, _) in ocr_result:
            detections.append(text)
            cv2.rectangle(rotated_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(rotated_cv2, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

    img_rgb = cv2.cvtColor(rotated_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    update_display_image(img_pil)

    output_textbox.delete(1.0, tk.END)
    if detections:
        output_textbox.insert(tk.END, "\n".join(detections))
    else:
        output_textbox.insert(tk.END, "No text detected.")
    output_text.clear()
    output_text.extend(detections)

# Save to CSV
def save_to_csv():
    content = output_textbox.get(1.0, tk.END).strip().splitlines()
    if not content:
        messagebox.showwarning("Warning", "No text to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Detected Text"])
            for line in content:
                writer.writerow([line])
        messagebox.showinfo("Success", "Saved to CSV.")

# Save to JSON
def save_to_json():
    content = output_textbox.get(1.0, tk.END).strip().splitlines()
    if not content:
        messagebox.showwarning("Warning", "No text to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON Files", "*.json")])
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({"detected_text": content}, f, indent=4)
        messagebox.showinfo("Success", "Saved to JSON.")

# GUI setup
root = tk.Tk()
root.title("YOLO + OCR Text Detector")
root.geometry("1200x800")

# Top buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Load Model", command=load_model).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Load Image", command=load_image).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Run YOLO + OCR", command=run_detection).grid(row=0, column=2, padx=5)
tk.Button(btn_frame, text="Save to CSV", command=save_to_csv).grid(row=0, column=3, padx=5)
tk.Button(btn_frame, text="Save to JSON", command=save_to_json).grid(row=0, column=4, padx=5)

# Image operation buttons
img_ops = tk.Frame(root)
img_ops.pack(pady=5)

tk.Button(img_ops, text="Zoom In", command=zoom_in).grid(row=0, column=0, padx=5)
tk.Button(img_ops, text="Zoom Out", command=zoom_out).grid(row=0, column=1, padx=5)
tk.Button(img_ops, text="Rotate Left", command=rotate_left).grid(row=0, column=2, padx=5)
tk.Button(img_ops, text="Rotate Right", command=rotate_right).grid(row=0, column=3, padx=5)

# Confidence slider
slider_frame = tk.Frame(root)
slider_frame.pack()
tk.Label(slider_frame, text="Confidence Threshold:").pack(side=tk.LEFT, padx=10)
threshold_slider = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL)
threshold_slider.set(30)
threshold_slider.pack()

# Image panel
panel = tk.Label(root)
panel.pack(pady=10)

# Output text
tk.Label(root, text="Detected Text (Editable):").pack()
output_textbox = ScrolledText(root, height=10, width=100)
output_textbox.insert(tk.END, "Detected text will appear here after running YOLO + OCR.")
output_textbox.pack(pady=10)

# Launch GUI
root.mainloop()
