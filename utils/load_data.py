import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        list: List of dictionaries representing the rows in the CSV file.
    """
    """
    Charge et nettoie le dataset.
    """
    data = pd.read_csv(file_path)
    return data
