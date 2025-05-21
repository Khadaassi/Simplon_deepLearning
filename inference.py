from tabulate import tabulate
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.build_pipeline import process_new_data_for_inference

def load_new_data(file_path):
    df = pd.read_csv(file_path)
    return df

def predict(file_path, model_path="best_model.keras"):
    print("Chargement du modèle...")
    model = load_model(model_path)

    print("Chargement et prétraitement des données...")
    new_data = load_new_data(file_path)
    X_new = process_new_data_for_inference(new_data)

    print("Prédiction en cours...")
    preds = model.predict(X_new)
    classes = np.argmax(preds, axis=1)
    proba = preds[:, 1] if preds.shape[1] > 1 else preds[:, 0]

    results = new_data.copy()
    results["Churn_Prob"] = proba
    results["Churn_Pred"] = classes


    # Traduction des prédictions (optionnel pour la lisibilité)
    results["Réponse"] = results["Churn_Pred"].map({0: "Va quitter", 1: "Reste fidèle"})

    # Format des colonnes à afficher
    table = results[["customerID", "Churn_Prob", "Réponse"]]
    print("\nRésultats de la prédiction :\n")
    print(tabulate(table, headers="keys", tablefmt="fancy_grid", showindex=False))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction du churn sur de nouvelles données")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier CSV des nouvelles données")
    parser.add_argument("--model", type=str, default="best_model.keras", help="Chemin vers le modèle entraîné")

    args = parser.parse_args()
    predict(args.input, args.model)
