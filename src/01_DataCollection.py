import pandas as pd
import logging
import os
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_safe_path(base_path, path, follow_symlinks=True):
    """Check if the path is safe to open.

    Args:
        base_path (str): The base directory against which to check the path.
        path (str): The path to check.
        follow_symlinks (bool): Whether to follow symlinks.

    Returns:
        bool: True if the path is safe, False otherwise.
    """
    # Resolve to absolute paths
    base_path = Path(base_path).resolve()
    target_path = Path(path).resolve()

    # Check if the target path is within the base path
    return base_path in target_path.parents or target_path == base_path

def validate_csv_file_path(file_path):
    """Validates the given file path to ensure it points to a CSV file and is safe.
    
    Args:
        file_path (str): The file path to validate.
        
    Returns:
        bool: True if the file path is valid and safe, False otherwise.
    """
    if not file_path.endswith('.csv'):
        logging.error("Invalid file format. File must be a CSV.")
        return False

    base_path = "../data/raw/diabetes.csv"
    if not is_safe_path(base_path, file_path):
        logging.error(f"Access to the path is not allowed: {file_path}")
        return False

    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False

    return True

def load_diabetes_data(file_path):
    """Loads the Pima Indians Diabetes dataset from the given file path.
    
    Args:
        file_path (str): The path to the CSV file containing the dataset.
        
    Returns:
        pandas.DataFrame: The loaded dataset, or None if an issue occurs.
    """
    if not validate_csv_file_path(file_path):
        return None
    
    try:
        df = pd.read_csv(file_path)
        logging.info("Dataset loaded successfully.")
        return df
    except pd.errors.ParserError as pe:
        logging.error(f"Error parsing CSV file: {pe}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {str(e)}")
        return None

def display_dataset_information(dataframe):
    """Displays information about the dataset.
    
    Args:
        dataframe (pandas.DataFrame): The dataset to display information for.
    """
    if dataframe is not None:
        print(dataframe.info())
        print(dataframe.describe())
        print(dataframe.isnull().sum())
        print(dataframe.head())
    else:
        logging.warning("No dataset to display information for.")

if __name__ == "__main__":
    data_path = "../data/raw/diabetes.csv"
    diabetes_data = load_diabetes_data(data_path)
    display_dataset_information(diabetes_data)
