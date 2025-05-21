import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def build_pipeline(dataframe: pd.DataFrame):
    y = dataframe['Churn'].map({"Yes": 0, "No": 1})
    dataframe = dataframe.drop(columns=["Churn"])

    binary_cols = [
        c for c in dataframe.columns
        if set(dataframe[c].dropna().unique()) <= {"Yes", "No"}
    ]

    multi_cat_cols = [
        c for c in dataframe.select_dtypes(include="object").columns
        if c not in binary_cols
    ]

    numeric_cols = dataframe.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train_0, X_test, y_train_0, y_test = train_test_split(dataframe, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_0, y_train_0, test_size=0.2, random_state=42, stratify=y_train_0)

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["No", "Yes"]]*len(binary_cols)))
    ])

    multi_cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("bin", binary_pipeline, binary_cols),
        ("cat", multi_cat_pipeline, multi_cat_cols),
        ("num", numeric_pipeline, numeric_cols)
    ])

    preprocessor.fit(X_train)

    X_train_preprocessed = preprocessor.transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test, preprocessor

def process_new_data_for_inference(new_data: pd.DataFrame):
    """
    Charge le pipeline sauvegardé et applique la transformation aux nouvelles données.
    """
    if "Churn" in new_data.columns:
        new_data = new_data.drop(columns=["Churn"])
    if "customerID" in new_data.columns:
        new_data = new_data.drop(columns=["customerID"])

    preprocessor = joblib.load("preprocessor.joblib")
    return preprocessor.transform(new_data)
