from ultralytics import YOLO

# Load a YOLO model (you can use yolov8n.pt, yolov5s.pt, etc.)
model = YOLO("yolov8n.pt")  # Or yolov5s.pt if using YOLOv5

# Run inference on an image
results = model("/home/pi/Downloads/horse.jpeg")

# Show results (for the first image in case of batch)
results[0].show()

# Optional: Print the detected objects
print(results[0].boxes.cls)         # Class IDs
print(results[0].boxes.conf)        # Confidence scores
print(results[0].boxes.xyxy) 
