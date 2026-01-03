import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "models", "xray_mri_model.h5")

# Load model ONCE
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (change according to your training)
class_names = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    
]


def predict_disease(image):
    # Convert to RGB and resize
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    predictions = model.predict(image)[0]  # flatten output

    # Get main disease & confidence
    class_index = np.argmax(predictions)
    confidence = round(float(predictions[class_index]) * 100, 2)
    disease = class_names[class_index]

    # -----------------------
    # NEW: define detailed outputs
    # -----------------------

    # All class probabilities
    probabilities = {class_names[i]: round(float(predictions[i]) * 100, 2) 
                     for i in range(len(class_names))}

    # Severity (simple logic)
    if confidence > 90:
        severity = "High"
    elif confidence > 70:
        severity = "Moderate"
    else:
        severity = "Low"

    # Explanation text
    explanation = (
        f"The AI model analyzed patterns in the X-ray/MRI image. "
        f"The most likely condition is *{disease}*, "
        f"with a confidence of {confidence}%. "
        "Please consult a doctor for accurate diagnosis."
    )

    return disease, confidence, severity, probabilities, explanation