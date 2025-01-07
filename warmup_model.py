import tensorflow as tf
import numpy as np
import os
from config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

# Path to save the warmed-up model

def configure_gpu():
    """
    Configure TensorFlow to use GPU with memory growth.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPUs available: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPUs found. Using CPU.")

def get_model():
    """
    Returns the warmed-up model. If not already warmed up, warms up the model and saves it.
    """
    

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Warm-up process
    print("Warming up the model...")
    dummy_image = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 3).astype(np.float32)
    with tf.device('/GPU:0'):
        model.predict(dummy_image)
    print("Warm-up completed.")

    # Save the warmed-up model to disk

    return model

# GPU configuration setup
configure_gpu()
