import argparse
import cv2
from tensorflow.keras.models import load_model
from utilis import preprocess_image
import numpy as np

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Define constants
MODEL_PATH = "saved_model/disaster_model.h5"
IMG_SIZE = (224, 224)
# Make sure class_labels are in the same order as during training
CLASS_LABELS = ['Damaged_Infrastructure', 'Fire_Disaster', 'Human_Damage', 'Land_Disaster', 'Non_Damage', 'Water_Disaster']

# Load the trained model
print(f"[INFO] loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Load and preprocess the image
image = cv2.imread(args["image"])
processed_image = preprocess_image(image, target_size=IMG_SIZE)

# Make a prediction
print("[INFO] classifying image...")
predictions = model.predict(processed_image)
prediction_idx = np.argmax(predictions, axis=1)[0]
confidence = predictions[0][prediction_idx]
predicted_label = CLASS_LABELS[prediction_idx]

# Display the result
print(f"Prediction: {predicted_label}")
print(f"Confidence: {confidence:.2%}")