# import the necessary packages
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import matplotlib.pyplot as plt

def preprocess_image(image, target_size):
    """
    Preprocesses an image for model input.

    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): The target size for the image (height, width).

    Returns:
        The preprocessed image.
    """
    # Resize the image
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)

    # Convert the image to a numpy array and expand dimensions
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Preprocess the image for the model
    image = imagenet_utils.preprocess_input(image)

    return image

def plot_history(history, output_path):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history: The training history object from Keras.
        output_path (str): The path to save the plot.
    """
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(output_path)

def show_gradcam():
    """
    Generates and displays a Grad-CAM heatmap.
    (Placeholder for now)
    """
    print("Grad-CAM functionality to be implemented.")