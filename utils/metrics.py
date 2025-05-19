
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import numpy as np

# fonction pour afficher les metriques
def metriques(model, X_test, y_test) :
    """
    Calcule et affiche les métriques de régression.

    Cette fonction calcule plusieurs métriques d'évaluation pour évaluer la performance d'un modèle de régression : 
    - MSE (Mean Squared Error) : Erreur quadratique moyenne.
    - RMSE (Root Mean Squared Error) : Racine carrée de l'erreur quadratique moyenne.
    - MAE (Mean Absolute Error) : Erreur absolue moyenne.
    - R² (Coefficient de détermination) : Mesure la proportion de variance expliquée par le modèle.
    - MedAE (Median Absolute Error) : Médiane des erreurs absolues.

    Args:
        model: Le modèle de régression déjà entraîné.
        X_test: Les caractéristiques des données de test.
        y_test: La variable cible correspondante pour les données de test.

    Returns:
        tuple: Contient les valeurs des métriques calculées (MSE, MAE, R², MedAE).
    """

    pred_y = model.predict(X_test)
    mse = mean_squared_error(y_test, pred_y)
    mae = mean_absolute_error(y_test, pred_y)
    r2 = r2_score(y_test, pred_y)
    med_ae = median_absolute_error(y_test, pred_y)

    print(f"MSE : {mse}")
    print(f"RMSE : {np.sqrt(mse)}")
    print(f"MAE : {mae}")
    print(f"R2 : {r2}")
    print(f"MedAE : {med_ae}")

    return mse, mae, r2, med_ae