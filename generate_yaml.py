import generate_yaml
import os
import yaml
# --- SET YOUR PATHS HERE ---
# Use the full path to the folder containing your 'train' and 'test' folders
dataset_root = os.path.abspath("./Dataset") 

data_config = {
    # The root path
    'path': dataset_root,
    
    # subfolder datasets
    'train': 'images', #training dataset
    'val': 'images_test',    #validation dataset
    
    'nc': 5,
    'names': [
        'Car', 
        'Person', 
        'Bus', 
        'Truck', 
        'Motorcycle'
    ]
}

try:
    with open('data.yaml', 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
    print(f"Successfully created data.yaml at {os.getcwd()}")
except Exception as e:
    print(f"An error occurred while writing the file: {e}")

print("\nIMPORTANT: Your dataset path is set to:", dataset_root)