import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import subprocess
import os

class YOLO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Detection GUI")

        # Variables
        self.model_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.input_mode = tk.StringVar(value="webcam")

        # Layout
        self.setup_widgets()

    def setup_widgets(self):
        # Model Selection
        model_frame = ttk.LabelFrame(self.root, text="Select YOLOv8 Model")
        model_frame.pack(fill="x", padx=10, pady=5)
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).pack(side="left", padx=5, pady=5)
        ttk.Button(model_frame, text="Browse", command=self.browse_model).pack(side="left", padx=5)
        ttk.Button(model_frame, text="Save Text", command=self.save_text).pack(padx=5, pady=5, anchor="e")

        # Input Mode
        input_frame = ttk.LabelFrame(self.root, text="Input Method")
        input_frame.pack(fill="x", padx=10, pady=5)

        ttk.Radiobutton(input_frame, text="Webcam", variable=self.input_mode, value="webcam", command=self.toggle_input).pack(side="left", padx=10)
        ttk.Radiobutton(input_frame, text="Image", variable=self.input_mode, value="image", command=self.toggle_input).pack(side="left")

        self.image_entry = ttk.Entry(input_frame, textvariable=self.image_path, width=50, state="disabled")
        self.image_entry.pack(side="left", padx=5)
        self.image_button = ttk.Button(input_frame, text="Browse Image", command=self.browse_image, state="disabled")
        self.image_button.pack(side="left", padx=5)

        # Run Button
        ttk.Button(self.root, text="Run Detection", command=self.run_detection).pack(pady=10)

        # Output Display
        output_frame = ttk.LabelFrame(self.root, text="Detection Output")
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.output_canvas = tk.Canvas(output_frame, height=400, bg="gray")
        self.output_canvas.pack(fill="both", expand=True)

        # Edit Button Placeholder
        ttk.Button(self.root, text="Edit Detections (Coming Soon)", state="disabled").pack(pady=5)
        # Detected Text Output Field
        text_frame = ttk.LabelFrame(self.root, text="Recognized Text")
        text_frame.pack(fill="x", padx=10, pady=5)
        self.detected_text = tk.StringVar()
        self.text_output = ttk.Entry(text_frame, textvariable=self.detected_text, width=80)

        self.text_output.pack(padx=5, pady=5)

    def save_text(self):
        text = self.detected_text.get()
        if not text.strip():
            messagebox.showwarning("No Text", "There is no detected text to save.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Detected Text"
        )
        if save_path:
            try:
                with open(save_path, "w") as file:
                    file.write(text)
                messagebox.showinfo("Success", f"Text saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save text:\n{str(e)}")

    def browse_model(self):
        filename = filedialog.askopenfilename(filetypes=[("YOLOv8 Models", "*.pt")])
        if filename:
            self.model_path.set(filename)

    def browse_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if filename:
            self.image_path.set(filename)

    def toggle_input(self):
        is_image = self.input_mode.get() == "image"
        state = "normal" if is_image else "disabled"
        self.image_entry.configure(state=state)
        self.image_button.configure(state=state)

    def run_detection(self):
        if not self.model_path.get():
            messagebox.showerror("Error", "Please select a model file.")
            return


        command = ["python", os.path.abspath("program_5.py"), "--model", os.path.abspath(self.model_path.get()),"--image",os.path.abspath(self.image_path.get())]


        if self.input_mode.get() == "image":
            if not self.image_path.get() or not os.path.exists(self.image_path.get()):
                messagebox.showerror("Error", "Please select a valid image.")
                return
            command.extend(["--image", self.image_path.get()])
        else:
            # Webcam mode – no additional flags
            pass

        # Run subprocess
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)

            # Extract detected text from output
            detected_output = result.stdout.strip()
            if detected_output:
                self.detected_text.set(detected_output)
            else:
                self.detected_text.set("No text detected.")

            self.display_latest_output()
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Failed to run detection script.")

    def display_latest_output(self):
        output_dir = "output"
        if not os.path.exists(output_dir):
            messagebox.showinfo("No Output", "No output folder found.")
            return

        # Get most recent filtered image
        files = [f for f in os.listdir(output_dir) if f.endswith("_conf.jpg")]
        if not files:
            messagebox.showinfo("No Detections", "No detection results found.")
            return

        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(output_dir, f)))
        img_path = os.path.join(output_dir, latest_file)

        # Show image
        img = Image.open(img_path)
        img.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img)
        self.output_canvas.delete("all")
        self.output_canvas.create_image(10, 10, anchor="nw", image=img_tk)
        self.output_canvas.image = img_tk  # Prevent garbage collection


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLO_GUI(root)
    root.mainloop()

