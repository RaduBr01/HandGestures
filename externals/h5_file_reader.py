import h5py

# Path to the .h5 file
file_path = "../training_data/asl_single_gesture_model.h5"

# Open the .h5 file in read mode
with h5py.File(file_path, 'r') as file:
    # Print all the keys in the .h5 file (i.e., datasets, groups)
    print("Keys in the .h5 file:")
    for key in file.keys():
        print(key)

    # Accessing the 'model_weights' group
    if 'model_weights' in file:
        model_weights_group = file['model_weights']
        print("\nContents of 'model_weights' group:")

        # Iterate through all the datasets (members) in the 'model_weights' group
        for weight_name in model_weights_group:
            weight_data = model_weights_group[weight_name]
            print(f"Weight name: {weight_name}")

            # Check if the weight is a dataset or a group
            if isinstance(weight_data, h5py.Dataset):
                # If it's a dataset, print its shape and data
                print(f"Shape of weight data: {weight_data.shape}")
                print(f"Weight data: {weight_data[:]}")
            elif isinstance(weight_data, h5py.Group):
                # If it's a group, list its subgroups or datasets
                print(f"'{weight_name}' is a group, containing the following:")
                for subgroup_name in weight_data:
                    print(f"  - {subgroup_name}")
            print("-" * 50)  # Separator between each weight's data
    else:
        print("No 'model_weights' group found.")