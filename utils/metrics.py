import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score, recall_score

def metrics(model, X_test, y_test, y_test_cat) -> None :
    
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nAccuracy sur le test set : {test_acc:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred, target_names=['yes','no']))
