from ultralytics import YOLO

# Initialize YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")

# Train on custom dataset
model.train(
    data="custom_dataset_hd",  # dataset path
    epochs=5,
    imgsz=224,
    batch=16
)
