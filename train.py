import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import build_model
from utilis import plot_history
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Define constants
DATASET_PATH = "Cyclone_Wildfire_Flood_Earthquake_Dataset"
TRAIN_PATH = os.path.join("dataset_structured", "train")
VAL_PATH = os.path.join("dataset_structured", "validation")
NUM_CLASSES = 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "saved_model/disaster_model.h5"
PLOT_PATH = "training_plot.png"

# Create data generators with augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # Increased rotation
    width_shift_range=0.2,  # Increased shift
    height_shift_range=0.2, # Increased shift
    shear_range=0.3,      # Increased shear
    zoom_range=0.3,       # Increased zoom
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # Added brightness augmentation
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data iterators
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Build the model
model = build_model(num_classes=NUM_CLASSES)

# Train the model
print("[INFO] training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights
)

# Save the model
print(f"[INFO] saving model to {MODEL_PATH}...")
if not os.path.exists("saved_model"):
    os.makedirs("saved_model")
model.save(MODEL_PATH)

# Plot the training history
print(f"[INFO] generating plot and saving to {PLOT_PATH}...")
plot_history(history, PLOT_PATH)

print("[INFO] training complete.")