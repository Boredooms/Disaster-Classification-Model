
import os
import shutil
import random
import glob
from PIL import Image

def clean_dataset(dataset_path):
    """
    Removes non-image and corrupted files from the dataset directory.
    """
    print(f"[INFO] Cleaning dataset at {dataset_path}...")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Attempt to open the image file
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it is, in fact, an image
            except (IOError, SyntaxError) as e:
                print(f"[INFO] Deleting corrupted or non-image file: {file_path}")
                os.remove(file_path)

def split_dataset(source_dir, base_dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a dataset of images into training, validation, and test sets.

    Args:
        source_dir (str): The path to the source directory containing class subfolders.
        base_dest_dir (str): The path to the destination directory where train/val/test folders will be created.
        train_ratio (float): The proportion of images to allocate to the training set.
        val_ratio (float): The proportion of images to allocate to the validation set.
        test_ratio (float): The proportion of images to allocate to the test set.
    """
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        return

    # Ensure the ratios sum to 1
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8:
        print("[ERROR] Ratios must sum to 1.")
        return

    # Get the class names from the subdirectories in the source directory
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"[INFO] Found classes: {class_names}")

    # Create train, validation, and test directories
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(base_dest_dir, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
        # Create class subdirectories within each split
        for class_name in class_names:
            class_path = os.path.join(split_path, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)

    # Process each class
    for class_name in class_names:
        print(f"[INFO] Processing class: {class_name}")
        class_source_path = os.path.join(source_dir, class_name)
        
        # Get all image files (jpg, jpeg, png)
        image_files = glob.glob(os.path.join(class_source_path, '*.jpg'))
        image_files.extend(glob.glob(os.path.join(class_source_path, '*.jpeg')))
        image_files.extend(glob.glob(os.path.join(class_source_path, '*.png')))
        
        random.shuffle(image_files)
        
        # Calculate split indices
        total_images = len(image_files)
        train_end = int(total_images * train_ratio)
        val_end = train_end + int(total_images * val_ratio)
        
        # Get the lists of files for each split
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Function to copy files
        def copy_files(files, split_name):
            dest_dir = os.path.join(base_dest_dir, split_name, class_name)
            for f in files:
                shutil.copy(f, dest_dir)

        # Copy files to their respective directories
        copy_files(train_files, 'train')
        copy_files(val_files, 'validation')
        copy_files(test_files, 'test')
        
        print(f"[INFO] ... {len(train_files)} images for training.")
        print(f"[INFO] ... {len(val_files)} images for validation.")
        print(f"[INFO] ... {len(test_files)} images for testing.")

if __name__ == '__main__':
    SOURCE_DATASET_DIR = 'Cyclone_Wildfire_Flood_Earthquake_Dataset'
    STRUCTURED_DATASET_DIR = 'dataset_structured'
    
    # Check if the structured dataset already exists
    if os.path.exists(STRUCTURED_DATASET_DIR):
        print(f"[INFO] Deleting existing structured dataset: {STRUCTURED_DATASET_DIR}")
        shutil.rmtree(STRUCTURED_DATASET_DIR)

    print("[INFO] Creating structured dataset...")
    split_dataset(SOURCE_DATASET_DIR, STRUCTURED_DATASET_DIR)
    print("[INFO] Dataset splitting complete.")
    
    print("[INFO] Cleaning structured dataset...")
    clean_dataset(STRUCTURED_DATASET_DIR)
    print("[INFO] Dataset cleaning complete.")