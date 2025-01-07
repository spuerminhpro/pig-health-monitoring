import os

# Base Directory for the Streamlit App
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO Model Configuration
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov9t_100_epochs.pt")

# Directory to Save Cropped Detections
OUTPUT_PATH = os.path.join(BASE_DIR, "output")

# Classification Model Configuration
MODEL_PATH = os.path.join(BASE_DIR, "models", "healthy_V5_224.keras")

# Image Preprocessing Settings
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Ensure necessary directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
