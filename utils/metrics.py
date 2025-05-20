from sklearn.metrics import roc_auc_score, f1_score, recall_score

def metrics(model, X_test, y_test) :
    
    pred_y = model.predict(X_test)
    pred_y_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, pred_y_proba)
    f1 = f1_score(y_test, pred_y)
    recall = recall_score(y_pred=pred_y, y_true=y_test)

    print(f"ROC AUC  : {roc_auc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Recall   : {recall:.4f}")

    return roc_auc, f1, recall
