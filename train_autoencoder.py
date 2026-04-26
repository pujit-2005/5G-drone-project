import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.models import Model
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. SETUP & CONFIGURATION ---
NOISY_IMAGES_PATH = 'data/noisy/'
CLEAN_IMAGES_PATH = 'data/clean/'
MODEL_SAVE_PATH = 'denoising_autoencoder.h5'
IMAGE_SIZE = (256, 256)

# --- 2. DATA LOADING & PREPROCESSING ---
print("Loading and preprocessing data...")

def load_images(folder_path, image_size):
    """Loads, resizes, and normalizes images from a folder."""
    # Using glob to handle potential missing trailing slash
    image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + \
                  glob.glob(os.path.join(folder_path, '*.png'))
    images = []
    for file in sorted(image_files): # Sort to ensure alignment
        img = cv2.imread(file)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img.astype('float32') / 255.0  # Normalize to [0, 1]
            images.append(img)
    return np.array(images)

# Load your datasets
# x_train will be the input (noisy images)
x_train_noisy = load_images(NOISY_IMAGES_PATH, IMAGE_SIZE)
# y_train will be the target (clean images)
y_train_clean = load_images(CLEAN_IMAGES_PATH, IMAGE_SIZE)

# Basic validation
if len(x_train_noisy) == 0 or len(y_train_clean) == 0:
    print("Error: No images found. Check your folder paths and image extensions (.jpg, .png).")
    exit()
if len(x_train_noisy) != len(y_train_clean):
    print("Error: The number of noisy images does not match the number of clean images.")
    exit()

print(f"Data loaded successfully: {len(x_train_noisy)} image pairs.")

# --- 3. MODEL ARCHITECTURE ---
print("Building the autoencoder model...")

def build_autoencoder(input_shape=(256, 256, 3)):
    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder

model = build_autoencoder(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
model.summary()

# --- 4. COMPILE AND TRAIN ---
print("Compiling and training the model...")

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train_noisy, y_train_clean,
          epochs=50,          # Adjust as needed. Start with 50.
          batch_size=16,      # Adjust based on your PC/GPU memory.
          shuffle=True,
          validation_split=0.1) # Use 10% of data for validation during training

# --- 5. SAVE AND EVALUATE ---
print("Training complete. Saving the model...")
model.save(MODEL_SAVE_PATH)

print(f"Model saved to {MODEL_SAVE_PATH}")
print("Now, let's visualize the results on a few images...")

# Select a few images to test
num_test_images = 5
test_indices = np.random.choice(len(x_train_noisy), num_test_images, replace=False)
test_noisy = x_train_noisy[test_indices]
test_clean = y_train_clean[test_indices]

# Get the model's predictions
denoised_images = model.predict(test_noisy)

# Plot the results
plt.figure(figsize=(20, 6))
for i in range(num_test_images):
    # Noisy Input
    ax = plt.subplot(3, num_test_images, i + 1)
    plt.imshow(test_noisy[i])
    ax.set_title("Noisy Input")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Denoised Output
    ax = plt.subplot(3, num_test_images, i + 1 + num_test_images)
    plt.imshow(denoised_images[i])
    ax.set_title("Denoised Output")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Original Clean
    ax = plt.subplot(3, num_test_images, i + 1 + 2 * num_test_images)
    plt.imshow(test_clean[i])
    ax.set_title("Ground Truth")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()