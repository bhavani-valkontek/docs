import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import os
import time
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

class YOLOInferenceGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Object Detection")
        self.geometry("1400x1000")

        self.model = None
        self.model_path = None
        self.image_path = None
        self.current_image = None
        self.annotated_image = None
        self.selected_webcam_index = 0
        self.output_dir = "output/images"
        os.makedirs(self.output_dir, exist_ok=True)

        self.webcam_running = False
        self.capture_ready = False
        self.rotation_angle = 0
        self.zoom_factor = 1.0
        self.do_preprocessing = True

        self.default_preproc_values = {
            "clipLimit": 6.0,
            "tileGridSize": 8,
            "bilateral_d": 8,
            "bilateral_sigmaColor": 82,
            "bilateral_sigmaSpace": 70,
            "gamma": 6.5
        }

        self.create_gui_elements()

    def create_gui_elements(self):
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(padx=20, pady=10, fill="x")

        self.model_button = ctk.CTkButton(self.top_frame, text="\U0001F4C1 Choose Model", command=self.load_model)
        self.model_button.grid(row=0, column=0, padx=10)

        self.image_button = ctk.CTkButton(self.top_frame, text="\U0001F5BC Choose Image", command=self.load_image)
        self.image_button.grid(row=0, column=1, padx=10)

        self.webcam_button = ctk.CTkButton(self.top_frame, text="\U0001F4F7 Webcam Capture", command=self.select_webcam_index)
        self.webcam_button.grid(row=0, column=2, padx=10)

        self.capture_button = ctk.CTkButton(self.top_frame, text="\u2709\ufe0f Capture Image", command=self.capture_webcam_image)
        self.capture_button.grid(row=0, column=3, padx=10)
        self.capture_button.configure(state="disabled")

        self.cancel_webcam_button = ctk.CTkButton(self.top_frame, text="\u274C Cancel Webcam", command=self.stop_webcam, fg_color="red")
        self.cancel_webcam_button.grid(row=0, column=4, padx=10)
        self.cancel_webcam_button.grid_remove()

        self.conf_label = ctk.CTkLabel(self.top_frame, text="Confidence:")
        self.conf_label.grid(row=0, column=5, padx=(20, 5))

        self.conf_slider = ctk.CTkSlider(self.top_frame, from_=0, to=1, command=self.sync_conf_entry)
        self.conf_slider.set(0.25)
        self.conf_slider.grid(row=0, column=6, padx=5)

        self.conf_entry = ctk.CTkEntry(self.top_frame, width=60)
        self.conf_entry.insert(0, "0.25")
        self.conf_entry.grid(row=0, column=7, padx=5)
        self.conf_entry.bind("<KeyRelease>", self.sync_slider)

        self.run_button = ctk.CTkButton(self.top_frame, text="\u25B6\ufe0f Run Detection", fg_color="green", command=self.run_detection)
        self.run_button.grid(row=0, column=8, padx=10)

        self.rotate_button = ctk.CTkButton(self.top_frame, text="\u21bb Rotate 90\u00b0", command=self.rotate_image)
        self.rotate_button.grid(row=0, column=9, padx=10)

        self.zoom_slider = ctk.CTkSlider(self.top_frame, from_=0.1, to=2.0, number_of_steps=19, command=self.set_zoom)
        self.zoom_slider.set(1.0)
        self.zoom_slider.grid(row=0, column=10, padx=10)

        self.reset_annot_button = ctk.CTkButton(self.top_frame, text="Reset Annotations", command=self.reset_annotations)
        self.reset_annot_button.grid(row=0, column=11, padx=10)

        self.toggle_preproc = ctk.CTkCheckBox(self.top_frame, text="Enable Preprocessing", command=self.toggle_preprocessing)
        self.toggle_preproc.select()
        self.toggle_preproc.grid(row=0, column=12, padx=10)

        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.image_label = ctk.CTkLabel(self.image_frame, text="No image loaded")
        self.image_label.pack(expand=True)

        self.result_textbox = ctk.CTkTextbox(self, height=200, font=("Arial", 16))
        self.result_textbox.pack(pady=10, padx=20, fill="x")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            self.model_path = path
            self.model = YOLO(path).to(DEVICE)
            self.model_button.configure(text="\u2705 " + os.path.basename(path))

    def load_image(self):
        self.stop_webcam()
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.current_image = cv2.imread(path)
            self.annotated_image = self.current_image.copy()
            self.resize_image()
            self.show_image(self.current_image)

    def select_webcam_index(self):
        available = [i for i in range(5) if cv2.VideoCapture(i).read()[0]]
        if available:
            idx = simpledialog.askinteger("Webcam", f"Select webcam index {available}", minvalue=0, maxvalue=max(available))
            if idx is not None:
                self.selected_webcam_index = idx
                self.toggle_webcam()

    def toggle_webcam(self):
        if not self.webcam_running:
            self.webcam_running = True
            self.capture_ready = False
            self.cancel_webcam_button.grid()
            self.capture_button.configure(state="normal")
            threading.Thread(target=self.capture_webcam_loop, daemon=True).start()

    def stop_webcam(self):
        self.webcam_running = False
        self.cancel_webcam_button.grid_remove()
        self.capture_button.configure(state="disabled")

    def capture_webcam_loop(self):
        cap = cv2.VideoCapture(self.selected_webcam_index)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam")
            return
        while self.webcam_running:
            ret, frame = cap.read()
            if ret:
                self.current_image = frame
                self.annotated_image = frame.copy()
                if not self.capture_ready:
                    self.show_image(frame)
            time.sleep(0.03)
        cap.release()

    def capture_webcam_image(self):
        if self.webcam_running:
            self.capture_ready = True
            self.stop_webcam()
            self.show_image(self.current_image)

    def show_image(self, cv2_image):
        image = self.process_image_for_display(cv2_image)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo

    def process_image_for_display(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.rotation_angle:
            image = image.rotate(self.rotation_angle, expand=True)
        w, h = image.size
        image = image.resize((int(w * self.zoom_factor), int(h * self.zoom_factor)), Image.Resampling.LANCZOS)
        return image

    def rotate_image(self):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.show_image(self.current_image)

    def set_zoom(self, val):
        self.zoom_factor = float(val)
        self.show_image(self.current_image)

    def resize_image(self):
        try:
            w = simpledialog.askinteger("Resize Width", "Enter width:", initialvalue=self.current_image.shape[1])
            h = simpledialog.askinteger("Resize Height", "Enter height:", initialvalue=self.current_image.shape[0])
            if w and h:
                self.current_image = cv2.resize(self.current_image, (w, h))
        except:
            pass

    def reset_annotations(self):
        if self.annotated_image is not None:
            self.current_image = self.annotated_image.copy()
            self.show_image(self.current_image)
            self.result_textbox.delete("1.0", "end")

    def toggle_preprocessing(self):
        self.do_preprocessing = not self.do_preprocessing

    def sync_conf_entry(self, val):
        self.conf_entry.delete(0, "end")
        self.conf_entry.insert(0, f"{float(val):.2f}")

    def sync_slider(self, event):
        try:
            val = float(self.conf_entry.get())
            if 0 <= val <= 1:
                self.conf_slider.set(val)
        except ValueError:
            pass

    def preprocess_image(self, image):
        if not self.do_preprocessing:
            return image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.default_preproc_values["clipLimit"], tileGridSize=(self.default_preproc_values["tileGridSize"],)*2)
        clahe_applied = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(clahe_applied, self.default_preproc_values["bilateral_d"], self.default_preproc_values["bilateral_sigmaColor"], self.default_preproc_values["bilateral_sigmaSpace"])
        adaptive_thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 1), np.uint8)
        morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        gamma = self.default_preproc_values["gamma"]
        gamma_corrected = np.array(255 * (morphed / 255) ** gamma, dtype='uint8')
        return cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)

    def run_detection(self):
        if not self.model or self.current_image is None:
            messagebox.showerror("Error", "Model or image missing")
            return

        start = time.time()
        conf = float(self.conf_entry.get())
        image_to_detect = self.preprocess_image(self.current_image)
        results = self.model.predict(image_to_detect, conf=conf)[0]

        annotated = self.current_image.copy()
        labels = []

        boxes = sorted(results.boxes, key=lambda b: b.xyxy[0][1])  # Sort top to bottom
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{results.names[cls]} {conf:.2f}"
            labels.append(results.names[cls])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.annotated_image = annotated
        self.show_image(annotated)
        self.result_textbox.delete("1.0", "end")
        self.result_textbox.insert("end", "\n".join(labels))

if __name__ == "__main__":
    app = YOLOInferenceGUI()
    app.mainloop()