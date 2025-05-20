import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score

def metrics(model, X_test, y_test, y_test_cat):
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nAccuracy sur le test set : {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=['yes','no']))

    
    print(model.predict(X_test).shape)
    print(np.unique(y_test))  # Affiche l'ordre des classes

    # Probabilités pour la classe positive
    y_pred_proba = model.predict(X_test)[:, 1]
    
    y_pred_label = (y_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_label)

    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"F1-score : {f1:.4f}")

