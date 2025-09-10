
from flask import Flask, request, render_template, jsonify
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Define constants
MODEL_PATH = "saved_model/disaster_model.h5"
IMG_SIZE = (224, 224)
CLASS_LABELS = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']

# Load the trained model
print(f"[INFO] Loading model from {MODEL_PATH}...")
try:
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load the model: {e}")
    print("[ERROR] Please ensure the model file exists at the specified path and is a valid Keras model.")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read image into memory
            img_bytes = file.read()

            # Load and preprocess the image from memory
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            image = np.array(image)
            # Convert RGB to BGR
            image = image[:, :, ::-1].copy()

            if image is None:
                return jsonify({'error': 'Could not read the image file.'}), 400

            image = cv2.resize(image, IMG_SIZE)
            image = image.astype("float") / 255.0
            processed_image = np.expand_dims(image, axis=0)

            # Make a prediction
            predictions = model.predict(processed_image)
            prediction_idx = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][prediction_idx]
            predicted_label = CLASS_LABELS[prediction_idx]

            # Return prediction as JSON
            return jsonify({
                'prediction': predicted_label,
                'confidence': f"{(confidence * 100):.2f}"
            })

        except Exception as e:
            return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)