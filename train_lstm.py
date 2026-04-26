# train_lstm.py

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- 1. Configuration ---
FEATURES_DIR = 'features/'
MODEL_SAVE_PATH = 'lstm_temporal_model.h5'

# --- 2. Load the Feature Data ---
print("Loading feature data...")
X = [] # To store feature sequences
y = [] # To store labels (0 for healthy, 1 for stressed, etc.)

class_folders = sorted(glob.glob(os.path.join(FEATURES_DIR, '*')))
class_names = [os.path.basename(f) for f in class_folders]
label_map = {name: i for i, name in enumerate(class_names)}

print(f"Found classes: {label_map}")

for class_name, label in label_map.items():
    sequence_files = glob.glob(os.path.join(FEATURES_DIR, class_name, '*.npy'))
    for seq_file in sequence_files:
        sequence_features = np.load(seq_file)
        X.append(sequence_features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Convert labels to one-hot encoding (e.g., 0 -> [1, 0], 1 -> [0, 1])
y = to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

# --- 3. Build the LSTM Model ---
print("Building the LSTM model...")

# Get sequence length and feature count from the data shape
NUM_TIMESTEPS = X_train.shape[1]
NUM_FEATURES = X_train.shape[2]
NUM_CLASSES = y_train.shape[1]

model = Sequential()
model.add(LSTM(
    units=128,
    input_shape=(NUM_TIMESTEPS, NUM_FEATURES)
))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 4. Train the Model ---
print("Training the model...")
history = model.fit(
    X_train,
    y_train,
    epochs=50, # Adjust as needed
    batch_size=8,
    validation_data=(X_test, y_test)
)

# --- 5. Save the final model ---
print(f"Training complete. Saving model to {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)