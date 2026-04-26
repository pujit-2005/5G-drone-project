# extract_features.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
import glob

# --- 1. Configuration ---
DATASET_PATH = 'dataset/'
FEATURES_DIR = 'features/'
IMAGE_SIZE = (224, 224) # VGG16 requires this input size

# --- 2. Load Pre-trained Model (VGG16) ---
print("Loading pre-trained VGG16 model...")
# We use 'imagenet' weights. 'include_top=False' removes the final classification layer.
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
# We will take the output of the final pooling layer as our feature vector.
feature_extractor = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
print("Model loaded.")

# --- 3. Helper function for feature extraction ---
def extract_image_features(img_path):
    """Loads an image, preprocesses it, and extracts features."""
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the features, flatten them from a 4D tensor to a 1D vector
    features = feature_extractor.predict(x, verbose=0)
    return features.flatten()

# --- 4. Main script to process all sequences ---
if not os.path.exists(FEATURES_DIR):
    os.makedirs(FEATURES_DIR)

# Get the class folders (e.g., 'healthy', 'stressed')
class_folders = glob.glob(os.path.join(DATASET_PATH, '*'))

print(f"Found classes: {[os.path.basename(f) for f in class_folders]}")

for class_folder in class_folders:
    class_name = os.path.basename(class_folder)
    print(f"Processing class: {class_name}")

    # Create a corresponding folder in the features directory
    class_features_dir = os.path.join(FEATURES_DIR, class_name)
    if not os.path.exists(class_features_dir):
        os.makedirs(class_features_dir)

    # Get all the sequence folders for this class
    sequence_folders = glob.glob(os.path.join(class_folder, '*'))

    for seq_folder in sequence_folders:
        sequence_name = os.path.basename(seq_folder)
        print(f"  - Processing sequence: {sequence_name}")

        # Get all image paths in the sequence, and sort them chronologically
        image_paths = sorted(glob.glob(os.path.join(seq_folder, '*.jpg')))
        
        sequence_features = []
        for img_path in image_paths:
            features = extract_image_features(img_path)
            sequence_features.append(features)

        # Save the sequence of features as a single .npy file
        save_path = os.path.join(class_features_dir, sequence_name + '.npy')
        np.save(save_path, np.array(sequence_features))

print("\nFeature extraction complete!")