from picamera2 import picamera2

picam = Picamera2()
picam.start_preview()
picam.capture_file('test.jpg')
print("Image captured as test.jpg")

picam.stop_preview()
