import torch
from PIL import Image
import numpy as np
import cv2
from picamera2 import Picamera2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/teampi/yolov5/runs/train/exp6/weights/best.pt')

# Initialize the Pi Camera
picam = Picamera2()
picam.configure(picam.create_still_configuration())
picam.start()

def capture_image():
    # Capture image and save it
    image_path = "captured_image.jpg"
    picam.capture_file(image_path)
    print(f"Image captured and saved at {image_path}")
    return image_path

def run_detection(image_path):
    # Load and run detection
    results = model(image_path)
    
    # Show results
    results.show()  # Display with bounding boxes
    # Print detections
    print(results.pandas().xyxy[0])  # Dataframe of detections
    return results

if __name__ == "__main__":
    print("Capturing image...")
    captured_image = capture_image()
    print("Running YOLOv5 detection...")
    detection_results = run_detection(captured_image)
