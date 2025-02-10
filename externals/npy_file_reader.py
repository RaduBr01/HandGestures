import os
import numpy as np

# Define the relative path to the file
file_path = os.path.join('..', 'training_data', 'A', '12.npy')

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        # Load the .npy file
        data = np.load(file_path)
        print("Data successfully loaded: on file ", file_path)
        print(data)
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
