import numpy as np
import os

dataset_dir = "../training_data/A"
for label in os.listdir(dataset_dir):
    for file in os.listdir(dataset_dir):
        if file.endswith(".npy"):
            data = np.load(os.path.join(dataset_dir, file))
            print(f"{file}: {data.shape}")