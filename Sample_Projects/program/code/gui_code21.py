import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import cv2
import numpy as np
import csv
import os


class YOLO_OCR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO + OCR Text Detection GUI")

        self.model_path = None
        self.image_path = None
        self.confidence = tk.DoubleVar(value=0.5)
        self.zoom = tk.DoubleVar(value=1.0)

        self.image = None
        self.tk_image = None

        self.build_top_toolbar()
        self.build_center_image_display()
        self.build_bottom_editor()

    def build_top_toolbar(self):
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Load model
        tk.Button(toolbar, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        self.model_label = tk.Label(toolbar, text="Model: None")
        self.model_label.pack(side=tk.LEFT, padx=5)

        # Load image
        tk.Button(toolbar, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        self.image_label = tk.Label(toolbar, text="Image: None")
        self.image_label.pack(side=tk.LEFT, padx=5)

        # Confidence slider
        tk.Label(toolbar, text="Confidence").pack(side=tk.LEFT, padx=5)
        conf_slider = ttk.Scale(toolbar, from_=0.0, to=1.0, variable=self.confidence, command=self.update_conf_entry,
                                length=100)
        conf_slider.pack(side=tk.LEFT)
        self.conf_entry = tk.Entry(toolbar, width=4, textvariable=self.confidence)
        self.conf_entry.pack(side=tk.LEFT)

        # Zoom slider
        tk.Label(toolbar, text="Zoom").pack(side=tk.LEFT, padx=5)
        zoom_slider = ttk.Scale(toolbar, from_=0.5, to=3.0, variable=self.zoom, command=self.update_zoom_entry,
                                length=100)
        zoom_slider.pack(side=tk.LEFT)
        self.zoom_entry = tk.Entry(toolbar, width=4, textvariable=self.zoom)
        self.zoom_entry.pack(side=tk.LEFT)

        # Run Detection Button
        tk.Button(toolbar, text="Run Detection", command=self.run_detection).pack(side=tk.LEFT, padx=5)

    def build_center_image_display(self):
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<MouseWheel>", self.mouse_zoom)
        self.pan_start = None
        self.offset = [0, 0]

    def build_bottom_editor(self):
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        tk.Label(bottom_frame, text="Detected Text:").pack(side=tk.LEFT, padx=5)
        self.text_field = ScrolledText(bottom_frame, height=4, wrap=tk.WORD)
        self.text_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        tk.Button(bottom_frame, text="Save to CSV", command=self.save_to_csv).pack(side=tk.LEFT, padx=5)

    def load_model(self):
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Model", "*.pt")])
        if path:
            self.model_path = path
            self.model_label.config(text=f"Model: {os.path.basename(path)}")
            print(f"Model loaded: {path}")

    def load_image(self):
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.image_label.config(text=f"Image: {os.path.basename(path)}")
            self.display_image()

    def display_image(self):
        if not self.image_path:
            return

        img = cv2.imread(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        zoom_factor = self.zoom.get()
        w, h = int(img_pil.width * zoom_factor), int(img_pil.height * zoom_factor)
        img_resized = img_pil.resize((w, h))

        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(self.offset[0], self.offset[1], anchor="nw", image=self.tk_image)

    def update_conf_entry(self, val):
        self.conf_entry.delete(0, tk.END)
        self.conf_entry.insert(0, f"{float(val):.2f}")

    def update_zoom_entry(self, val):
        self.zoom_entry.delete(0, tk.END)
        self.zoom_entry.insert(0, f"{float(val):.2f}")
        self.display_image()

    def start_pan(self, event):
        self.pan_start = (event.x, event.y)

    def pan_image(self, event):
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.offset[0] += dx
        self.offset[1] += dy
        self.pan_start = (event.x, event.y)
        self.display_image()

    def mouse_zoom(self, event):
        if event.delta > 0:
            self.zoom.set(min(self.zoom.get() + 0.1, 3.0))
        else:
            self.zoom.set(max(self.zoom.get() - 0.1, 0.5))
        self.display_image()

    def run_detection(self):
        if not self.model_path or not self.image_path:
            print("Please load both model and image before running detection.")
            return

        # Placeholder for YOLO + OCR logic
        print(f"Running detection with model: {self.model_path} on image: {self.image_path}")

        # Example output
        dummy_text = "Detected text will appear here after running YOLO + OCR."
        self.text_field.delete("1.0", tk.END)
        self.text_field.insert(tk.END, dummy_text)

    def save_to_csv(self):
        text = self.text_field.get("1.0", tk.END).strip()
        if not text:
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if save_path:
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Detected Text"])
                writer.writerow([text])
            print(f"Saved detected text to {save_path}")


if __name__ == '__main__':
    root = tk.Tk()
    app = YOLO_OCR_GUI(root)
    root.mainloop()
