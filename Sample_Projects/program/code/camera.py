import cv2

# Initialize the camera (0 for default USB camera, use 1, 2, etc., if multiple cameras are connected)
cap = cv2.VideoCapture(1)

# Set camera resolution (adjust based on your camera capabilities)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
if not cap.isOpened():
    exit()

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Show the live camera feed
    cv2.imshow("Live Camera - Press 's' to Capture", frame)

    # Press 's' to capture and process the image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("./images/captured_image.jpg", frame)  # Save the captured image
        print("Image captured and saved as 'captured_image.jpg'")
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()