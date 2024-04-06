import cv2
from ultralytics import YOLO
import imutils
import time
import torch


# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model = YOLO('/home/hossein/PycharmProjects/yolo-linux/ultralytics/yolov8x-pose.pt').to(device)

# Open the Webcam
cap = cv2.VideoCapture(0)

# allow the camera or video file to warm up
time.sleep(2.0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # resize the frame
        frame = imutils.resize(frame, width=1000)
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()