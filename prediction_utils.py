import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# Preprocessing 
def preprocess_image(img_path, img_height=224, img_width=224):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Prediction 
def predict_with_model(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0]  
    confidence_healthy = 1 - prediction[0]  # Confidence for "Healthy"
    confidence_sick = prediction[0]  # Confidence for "Sick"
    predicted_class = "Sick" if confidence_sick > confidence_healthy else "Healthy"
    return predicted_class, confidence_healthy, confidence_sick