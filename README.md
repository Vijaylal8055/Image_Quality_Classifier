# Image_Quality_Classifier
Image Quality Classifier using YOLOv8
This project is an Image Quality Detection application built with YOLOv8 and Streamlit. The model classifies images into categories such as Normal, Blurred, Edge-cut, and Half-visible.
Features
Detects image quality issues in uploaded images
Classifies images into predefined categories
Interactive web interface using Streamlit
Model trained on custom dataset using YOLOv8 classification framework
Technologies Used
YOLOv8-for image classification
Streamlit-for UI
Pillow-for image handling
Python
How it Works
The YOLOv8 classification model is initialized with pretrained weights.
The model is fine-tuned on your custom dataset.
The Streamlit app loads the trained model and accepts image uploads.
The model predicts the class and displays the result in real-time.
Configuration
Model weights: runs/classify/train/weights/best.pt
Dataset path: Set in train_yolo.py (data="custom_dataset_hd")
Image size & batch size: Adjustable in the training script
Classes: Defined by the dataset folder names
Future Improvements
Expand dataset with more diverse image quality issues
Improve accuracy by tuning hyperparameters
Add deployment support (Heroku, AWS, etc.)
Integrate with image processing pipelines

