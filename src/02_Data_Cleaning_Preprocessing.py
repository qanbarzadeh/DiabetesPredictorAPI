import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='../config/config.json'):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return None

def load_data(file_path):
    """Loads data from the specified file path."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error("File is empty.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None

def preprocess_data(data):
    """Preprocesses the dataset."""
    # Identify and handle missing values
    columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)
    
    # Impute missing values using the median
    imputer = SimpleImputer(strategy='median')
    data[columns_with_zeros] = imputer.fit_transform(data[columns_with_zeros])
    
    # Feature scaling
    scaler = StandardScaler()
    columns_to_scale = data.columns.drop('Outcome')
    data_scaled = pd.DataFrame(scaler.fit_transform(data[columns_to_scale]), columns=columns_to_scale)
    data_scaled['Outcome'] = data['Outcome']
    
    return data_scaled

def save_data(data, file_path):
    """Saves the preprocessed data to the specified file path."""
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Data saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")

def main():
    config = load_config()
    if config is None:
        return
    
    data_path = Path(config['data_path'])
    processed_data_path = Path(config['processed_data_path'])
    
    diabetes_data = load_data(data_path)
    if diabetes_data is not None:
        preprocessed_data = preprocess_data(diabetes_data)
        save_data(preprocessed_data, processed_data_path)

if __name__ == "__main__":
    main()
