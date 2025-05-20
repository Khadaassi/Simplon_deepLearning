import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split


def build_pipeline(dataframe: pd.DataFrame) :
    
    y = dataframe['Churn'].map({"Yes": 1, "No": 0}) 
    dataframe = dataframe.drop(columns=["Churn"])


    binary_cols = [
        c for c in dataframe.columns
        if set(dataframe[c].dropna().unique()) <= {"Yes", "No"}
    ]

    # b) Colonnes catégorielles à > 2 modalités
    multi_cat_cols = [
        c for c in dataframe.select_dtypes(include="object").columns
        if c not in binary_cols
    ]

    # c) Colonnes numériques
    numeric_cols = dataframe.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_train_0, X_test, y_train_0, y_test = train_test_split(dataframe, y, stratify=y, test_size=0.2, random_state=42)

    # On prend 20% de X_train pour validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_0, y_train_0, test_size=0.2, random_state=42, stratify=y_train_0
    )

    # ------------------------------------------------------------------
    # Pipelines de pré-traitement
    # ------------------------------------------------------------------
    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["No", "Yes"]]*len(binary_cols)))
    ])

    # Pipeline pour les colonnes multi-catégorielles (>2 modalités)
    multi_cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    # Optional : colonnes numériques
    numeric_cols = dataframe.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ------------------------------------------------------------------
    # Assemblage
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer([
        ("bin", binary_pipeline, binary_cols),
        ("cat", multi_cat_pipeline, multi_cat_cols),
        ("num", numeric_pipeline, numeric_cols)
    ])

    # Entraîner le pipeline sur les données
    preprocessor.fit(X_train)

    # Transformation
    X_train_preprocessed = preprocessor.transform(X_train)
    X_val_preprocessed  = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)


        
    return X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test