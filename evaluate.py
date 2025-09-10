import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Define constants
DATASET_PATH = "dataset_structured"
TEST_PATH = os.path.join(DATASET_PATH, "test")
MODEL_PATH = "saved_model/disaster_model.h5"
NUM_CLASSES = 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Load the trained model
print(f"[INFO] loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# Evaluate the model on the test set
print("[INFO] evaluating model...")
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Get predictions
print("[INFO] generating predictions...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Print classification report
print("\nClassification Report:")
labels = np.arange(len(class_labels))
print(classification_report(y_true, y_pred, target_names=class_labels, labels=labels))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=labels))