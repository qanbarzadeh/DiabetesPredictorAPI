
import json
import os

# Configuration data to be written
config_data = {
    "data_path": "./data/raw/diabetes.csv",
    "processed_data_path": "./data/processed/diabetes_processed.csv"
}

# Specify the directory and file path
config_directory = 'config'
config_file_path = os.path.join(config_directory, 'config.json')

# Create the config directory if it does not exist
if not os.path.exists(config_directory):
    os.makedirs(config_directory)

# Write the configuration data to config.json
with open(config_file_path, 'w') as config_file:
    json.dump(config_data, config_file, indent=4)

print(f"Configuration file created at {config_file_path}")
