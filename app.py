from flask import Flask, render_template, request, url_for
import torch
from picamera2 import Picamera2

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/teampi/yolov5/runs/train/exp6/weights/best.pt')

# Initialize the Pi Camera
picam = Picamera2()
picam.configure(picam.create_still_configuration())
picam.start()

@app.route("/")
def index():
    return render_template("index.html", image_path=None)

@app.route("/capture", methods=["POST"])
def capture_and_detect():
    # Capture an image
    image_path = "static/captured_image.jpg"
    picam.capture_file(image_path)

    # Run YOLOv5 detection
    results = model(image_path)
    results.save(save_dir="static")  # Save detection result in static folder as detected_image.jpg

    # Redirect to the main page and show the result
    return render_template("index.html", image_path="detected_image.jpg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
