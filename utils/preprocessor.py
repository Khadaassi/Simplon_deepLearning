import pandas as pd
import numpy as np


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """preprocess_data permet de prétraiter les donnnées

    Args:
        dataframe (pd.DataFrame): donnée d'entrée

    Returns:
        dataFrame: donnée de sortie prétraitée 
    """
    
    dataframe["TotalCharges"] = dataframe["TotalCharges"].replace(" ", np.nan)
    dataframe["TotalCharges"] = dataframe["TotalCharges"].astype(float)
    
    dataframe = dataframe.drop(columns=["customerID"])

    return dataframe

