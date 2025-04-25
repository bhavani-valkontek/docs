import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import os
from datetime import datetime

# === Config ===
SAVE_FOLDER = "captured_images"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# === Global Variables ===
zoom = 1.0
contrast = 1.0
frame_width, frame_height = 640, 480

# === Camera Setup ===
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# === GUI Setup ===
root = tk.Tk()
root.title("USB Camera Preview")

# === Functions ===
def update_frame():
    global frame, zoom, contrast
    ret, frame = cap.read()
    if not ret:
        return

    # Apply contrast
    frame_adj = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

    # Apply zoom
    h, w = frame_adj.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius_x, radius_y = int(w / (2 * zoom)), int(h / (2 * zoom))
    min_x, max_x = center_x - radius_x, center_x + radius_x
    min_y, max_y = center_y - radius_y, center_y + radius_y
    cropped = frame_adj[min_y:max_y, min_x:max_x]
    frame_resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert to ImageTk
    img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

def zoom_in():
    global zoom
    zoom = min(zoom + 0.1, 3.0)

def zoom_out():
    global zoom
    zoom = max(zoom - 0.1, 1.0)

def contrast_up():
    global contrast
    contrast = min(contrast + 0.1, 3.0)

def contrast_down():
    global contrast
    contrast = max(contrast - 0.1, 0.1)

def capture_image():
    global frame
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    filepath = os.path.join(SAVE_FOLDER, filename)
    cv2.imwrite(filepath, frame)
    print(f"📸 Image saved: {filepath}")
    show_preview(filepath)

def show_preview(filepath):
    preview_win = tk.Toplevel(root)
    preview_win.title("Captured Image Preview")

    img = Image.open(filepath)
    img.thumbnail((600, 600))
    imgtk = ImageTk.PhotoImage(img)

    label = tk.Label(preview_win, image=imgtk)
    label.image = imgtk
    label.pack()

def mouse_wheel(event):
    if event.delta > 0:
        zoom_in()
    else:
        zoom_out()

# === Layout ===
video_label = Label(root)
video_label.pack()

controls = tk.Frame(root)
controls.pack()

Button(controls, text="Zoom +", command=zoom_in).grid(row=0, column=0)
Button(controls, text="Zoom -", command=zoom_out).grid(row=0, column=1)
Button(controls, text="Contrast +", command=contrast_up).grid(row=0, column=2)
Button(controls, text="Contrast -", command=contrast_down).grid(row=0, column=3)
Button(controls, text="Capture", command=capture_image).grid(row=0, column=4)

# Mouse wheel binding
video_label.bind("<MouseWheel>", mouse_wheel)

# Start
update_frame()
root.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
