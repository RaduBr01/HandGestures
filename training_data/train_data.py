import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Directory where the gesture data is stored (folders for A, B, C, etc.)
dataset_dir = os.getcwd()  # Using the current working directory

# Load the data for multiple gestures
def load_multiple_gesture_data():
    X = []
    y = []
    class_labels = {folder: idx for idx, folder in enumerate(os.listdir(dataset_dir)) if
                    os.path.isdir(os.path.join(dataset_dir, folder))}

    for folder, label in class_labels.items():
        folder_path = os.path.join(dataset_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                feature_vector = np.load(os.path.join(folder_path, file))
                X.append(feature_vector)
                y.append(label)  # Assign the class label based on the folder

    X = np.array(X)
    y = np.array(y)
    return X, y


# Load the data
X, y = load_multiple_gesture_data()


# Normalize the features (optional but recommended)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(np.unique(y)),"   ",  len(X))
# Build the model
model = Sequential([
    Dense(128, input_shape=(X.shape[1] ,), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(y), activation='softmax')  # Output layer for multi-class classification (3 classes)
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('new_model.keras')
print("Model trained and saved as 'new_model.keras'")
