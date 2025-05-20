import pandas as pd
import numpy as np


def preprocess_data(dataframe: pd.DataFrame):
    
    dataframe["TotalCharges"] = dataframe["TotalCharges"].replace(" ", np.nan)
    dataframe["TotalCharges"] = dataframe["TotalCharges"].astype(float)
    
    return dataframe

