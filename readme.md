# AI Disaster Classification Model

This project implements a deep learning model to classify images of natural disasters into four categories: Cyclone, Earthquake, Flood, and Wildfire. It leverages transfer learning by fine-tuning a pre-trained ResNet50 model.

## Project Structure

```
Ai-Disaster-Classification-Model/
├── Cyclone_Wildfire_Flood_Earthquake_Dataset/  # Raw image data
├── app.py                                      # Interactive prediction script
├── dataset_structured/                         # Organized train/val/test data
├── evaluate.py                                 # Model evaluation script
├── model.py                                    # Model architecture definition
├── predict.py                                  # Command-line prediction script
├── prepare_dataset.py                          # Data cleaning and splitting script
├── readme.md                                   # This file
├── requirements.txt                            # Project dependencies
├── saved_model/
│   └── disaster_model.h5                       # The trained model file
└── test/                                       # (Optional) test images
```

## Files

### `prepare_dataset.py`

**Purpose**: To take the raw, mixed dataset of images and structure it into organized directories for training, validation, and testing. This script also cleans the dataset by removing any corrupted or non-image files.

**Detailed Algorithm**:

1.  **Dataset Cleaning (`clean_dataset`)**:
    *   **Objective**: Ensure every file in the dataset is a valid, readable image to prevent errors during training.
    *   **Process**:
        *   It recursively walks through every file in the provided `dataset_path`.
        *   For each file, it uses the `PIL.Image.open()` function within a `try...except` block.
        *   If `PIL.Image.open()` succeeds, the file is a valid image, and it is closed immediately.
        *   If it fails and raises an exception (like `PIL.UnidentifiedImageError` or a generic `IOError`), the file is considered corrupt or not an image.
        *   The script then prints the path of the corrupt file and deletes it using `os.remove()`.

2.  **Dataset Splitting (`split_dataset`)**:
    *   **Objective**: To partition the cleaned dataset into three standard subsets:
        *   **Training set**: Used to train the model.
        *   **Validation set**: Used to monitor the model's performance on unseen data during training to tune hyperparameters and prevent overfitting.
        *   **Test set**: Used for the final, unbiased evaluation of the trained model's performance.
    *   **Process**:
        *   It first deletes the `base_dest_dir` (the `dataset_structured` folder) if it already exists, ensuring a fresh split every time.
        *   It iterates through each class directory (e.g., "Cyclone") in the `source_dir`.
        *   For each class, it creates corresponding subdirectories in the `train`, `validation`, and `test` folders.
        *   It lists all images for that class and shuffles them randomly using `random.shuffle()` to ensure the splits are not biased by the file order.
        *   It calculates the number of images for each split based on the provided ratios (`train_ratio`, `val_ratio`, `test_ratio`).
        *   It uses list slicing to partition the shuffled list of images.
        *   Finally, it iterates through these partitioned lists and copies each file from the source to its new destination (e.g., `dataset_structured/train/Cyclone/`) using `shutil.copy2()`.

### `model.py`

**Purpose**: To define the architecture of the Convolutional Neural Network (CNN). This script uses the powerful technique of **transfer learning**.

**Detailed Algorithm (Transfer Learning and Fine-Tuning)**:

1.  **Load Pre-trained Base Model**:
    *   **Model**: `ResNet50` (Residual Network with 50 layers).
    *   **Rationale**: Instead of training a very deep network from scratch, which requires a massive amount of data and computational power, we use a model that has already been trained on the ImageNet dataset. This pre-training has taught the model to recognize a vast library of low-level and mid-level features (edges, textures, patterns, simple shapes).
    *   **`include_top=False`**: This is a critical parameter. It means we are only loading the convolutional base of ResNet50 (the feature extraction part) and excluding its final fully connected layers (the original classification part for 1000 ImageNet classes).
    *   **`weights='imagenet'`**: This specifies that we want to load the weights learned from the ImageNet training.

2.  **Fine-Tuning Strategy**:
    *   **Concept**: We want to retain the general feature knowledge from the early layers of ResNet50 but adapt the more specialized, high-level features from the later layers to our specific disaster images.
    *   **Freezing Layers**: The first 143 layers of the ResNet50 model are "frozen" by setting `layer.trainable = False`. Their weights, learned from ImageNet, will not be changed during our training. This preserves the robust, general feature extractors.
    *   **Unfreezing (Thawing) Layers**: The layers from 143 onwards are left as `trainable`. Their weights will be slightly adjusted during our training process, allowing them to specialize in recognizing features relevant to cyclones, earthquakes, etc.

3.  **Building a Custom Classification Head**:
    *   **Objective**: To add new layers on top of the ResNet50 base that can take the extracted features and perform our specific 4-class classification.
    *   **`GlobalAveragePooling2D`**:
        *   This layer takes the final feature maps from the ResNet50 base (which are multi-dimensional tensors) and converts each feature map into a single number by taking its average.
        *   **Advantage**: It drastically reduces the number of parameters compared to a traditional `Flatten` layer, making the model less prone to overfitting.
    *   **`Dense(1024, activation='relu')`**: A standard fully connected layer with 1024 neurons. It acts as a classifier, learning complex combinations of the features provided by the pooling layer. The `relu` activation introduces non-linearity.
    *   **`Dropout(0.3)`**: A regularization technique. During each training step, it randomly deactivates 30% of the neurons in the previous layer. This prevents the model from becoming too reliant on any single neuron or feature, forcing it to learn more robust and generalizable patterns.
    *   **`Dense(512, activation='relu')`**: Another dense layer to further refine the classification logic.
    *   **`Dense(num_classes, activation='softmax')`**: The final output layer.
        *   It has a number of neurons equal to the number of classes (4 in our case).
        *   The **`softmax`** activation function is essential for multi-class classification. It takes the raw output scores (logits) and squashes them into a probability distribution that sums to 1. Each neuron's output is the model's predicted probability that the image belongs to that class.

4.  **Model Compilation**:
    *   **Objective**: To configure the model for training by specifying the optimizer, loss function, and metrics.
    *   **`optimizer=Adam(learning_rate=1e-5)`**:
        *   The `Adam` optimizer is an efficient and popular choice.
        *   A **very low learning rate** (`0.00001`) is crucial for fine-tuning. A high learning rate would risk making large updates to the weights of the unfrozen layers, potentially destroying the valuable pre-trained features.
    *   **`loss='categorical_crossentropy'`**: This is the standard loss function for multi-class classification. It measures how different the predicted probability distribution is from the true label. The model's goal is to minimize this loss.
    *   **`metrics=['accuracy']`**: We monitor the `accuracy` metric during training to see how the model is performing.

### `train.py`

**Purpose**: To execute the training process. It feeds the prepared data into the model and saves the resulting trained model.

**Detailed Algorithm**:

1.  **Data Augmentation (`ImageDataGenerator`)**:
    *   **Objective**: To artificially expand the training dataset and make the model more robust to variations in real-world images.
    *   **Process**: It defines a set of random transformations that will be applied to the training images on-the-fly.
        *   `rotation_range=30`: Randomly rotate images.
        *   `width_shift_range=0.2`, `height_shift_range=0.2`: Randomly shift images horizontally or vertically.
        *   `shear_range=0.2`: Randomly apply a shearing transformation.
        *   `zoom_range=0.2`: Randomly zoom in on images.
        *   `horizontal_flip=True`: Randomly flip images horizontally.
        *   `brightness_range=[0.8, 1.2]`: Randomly adjust image brightness.
    *   The validation data generator (`val_datagen`) only performs pixel scaling, as we want to evaluate the model on the original, untransformed validation images.

2.  **Data Generators (`flow_from_directory`)**:
    *   **Concept**: Instead of loading all images into memory at once (which would be impossible for large datasets), these generators load images in batches from their directories.
    *   **Process**: They automatically read images from the `train` and `validation` folders, resize them to `(224, 224)`, apply the defined augmentations (for training data), convert them to batches of tensors, and feed them to the model.

3.  **Class Weighting (`compute_class_weight`)**:
    *   **Objective**: To address class imbalance. If one class has significantly more images than another, the model might become biased.
    *   **Process**: It calculates weights that are inversely proportional to the class frequencies. Under-represented classes get a higher weight, which means the model will incur a larger penalty for misclassifying them, forcing it to pay more attention to those classes.

4.  **Model Training (`model.fit`)**:
    *   **Process**: This is the core training loop.
        *   It runs for a specified number of `epochs` (one epoch is one full pass over the entire training dataset).
        *   In each step of an epoch, the model receives a batch of images from the `train_generator`.
        *   It performs a **forward pass** (makes a prediction), calculates the **loss** (how wrong the prediction was), and then performs **backpropagation** to calculate how much each weight contributed to the error.
        *   The `Adam` optimizer then updates the model's trainable weights to reduce the loss.
        *   After each epoch, it evaluates the model on the `val_generator` to track its performance on unseen data.

5.  **Saving**:
    *   Once training is complete, `model.save()` serializes the model's architecture, weights, and optimizer state into a single HDF5 file (`.h5`).

### `evaluate.py`

**Purpose**: To provide a quantitative and unbiased assessment of the final trained model's performance on the test set.

**Detailed Algorithm**:

1.  **Load Model and Test Data**:
    *   It loads the saved `disaster_model.h5`.
    *   It creates a generator for the test set using `ImageDataGenerator` (only rescaling) and `flow_from_directory`. `shuffle=False` is important here to ensure the predictions align with the true labels.

2.  **Generate Predictions**:
    *   `model.predict()` is called on the test generator to get the model's output (probability distributions) for every image in the test set.
    *   `np.argmax(..., axis=1)` is used to find the index of the highest probability for each prediction, which corresponds to the predicted class label.

3.  **Calculate and Display Metrics**:
    *   **Classification Report**: This report from `scikit-learn` provides a breakdown of performance for each class using:
        *   **Precision**: Of all the images the model *predicted* as "Flood", what percentage were actually "Flood"? (`TP / (TP + FP)`)
        *   **Recall (Sensitivity)**: Of all the actual "Flood" images in the dataset, what percentage did the model correctly identify? (`TP / (TP + FN)`)
        *   **F1-Score**: The harmonic mean of precision and recall, providing a balanced measure.
    *   **Confusion Matrix**: This is a table that visualizes performance.
        *   The rows represent the true classes.
        *   The columns represent the predicted classes.
        *   The diagonal elements show the number of correct predictions for each class.
        *   Off-diagonal elements show where the model is making mistakes (e.g., the number of times it predicted "Cyclone" when the image was actually "Wildfire").

### `app.py` and `predict.py`

**Purpose**: To use the trained model for inference on new, single images.

**Detailed Algorithm**:

1.  **Load Model**: The saved `disaster_model.h5` is loaded.
2.  **Image Preprocessing**:
    *   **Objective**: Any new image must be transformed to exactly match the format of the images the model was trained on.
    *   **Process**:
        *   The image is loaded from its file path using `PIL.Image.open()`.
        *   It's resized to `(224, 224)` pixels.
        *   It's converted to a NumPy array of shape `(224, 224, 3)`.
        *   The pixel values are scaled from the `[0, 255]` integer range to the `[0, 1]` float range by dividing by 255.0.
        *   The array's dimensions are expanded using `np.expand_dims(..., axis=0)` to create a shape of `(1, 224, 224, 3)`. This is because the model is designed to work on batches of images, so we treat our single image as a batch of size 1.
3.  **Inference**:
    *   The preprocessed image array is passed to `model.predict()`.
    *   The model returns a 2D array containing the probability distribution for our single image (e.g., `[[0.1, 0.05, 0.8, 0.05]]`).
4.  **Output**:
    *   `np.argmax()` finds the index of the maximum value in the result (in the example above, index 2).
    *   This index is used to look up the corresponding class name from the list of classes.
    *   The predicted class and the confidence score (the probability value) are printed.

### `requirements.txt`

This file lists all the Python libraries (e.g., `tensorflow`, `numpy`, `scikit-learn`) required for the project. It allows for easy setup of the environment using the command `pip install -r requirements.txt`.