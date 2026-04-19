import torch
import cv2
import numpy as np

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.conf = 0.30  # Confidence threshold

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam opened successfully. Press 'q' to quit.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        # Run inference
        results = model(frame)

        # Get annotated frame (returns BGR format for OpenCV display)
        annotated_frame = np.squeeze(results.render())  # Already in BGR format

        # Display the resulting frame
        cv2.imshow("Surveillance to MIDI - Webcam", annotated_frame)

        # Print detection results (optional)
        detections = results.pandas().xyxy[0]
        if not detections.empty:
            print("Detections:")
            print(detections[["name", "confidence"]].to_string(index=False))

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
