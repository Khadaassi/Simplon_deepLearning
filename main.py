import datetime
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import tensorflow as tf
from utils.load_data import load_data
from utils.preprocessor import preprocess_data
from utils.build_pipeline import build_pipeline
from utils.metrics import metrics
from utils.build_model import build_model, get_class_weights, run_hyperparameter_search
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils.viz import matrix, plot_loss_acc
import json

data = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def one_hot_labels(y, num_classes=2):
    return tf.keras.utils.to_categorical(y, num_classes)

def main():
    USE_TUNER = True
    set_seed(42)
    original_data = preprocess_data(data)

    X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test = build_pipeline(original_data)

    num_classes = len(np.unique(y_train))

    y_train_cat = one_hot_labels(y_train, num_classes)
    y_val_cat = one_hot_labels(y_val, num_classes)
    y_test_cat = one_hot_labels(y_test, num_classes)

    class_weights = get_class_weights(y_train)

    if USE_TUNER:
        model, best_hps = run_hyperparameter_search(X_train_preprocessed, y_train_cat, X_val_preprocessed, y_val_cat, num_classes)
        print("Best hyperparameters:")
        print(best_hps.values)
        batch_size = 16#32  # Par défaut 32
    else:
        model = build_model(X_train_preprocessed, num_classes)
        batch_size = 16

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_cb = ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor='val_loss', mode='min'
    )

    early_stop_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_preprocessed, y_train_cat,
        validation_data=(X_val_preprocessed, y_val_cat),
        epochs=20, class_weight=class_weights,
        batch_size=batch_size,
        verbose=1,
        callbacks=[tensorboard_cb, checkpoint_cb, early_stop_cb]
    )

    print("\nEvaluation du modèle Deep Learning :")
    y_pred_dl = metrics(model, X_test_preprocessed, y_test, y_test_cat)

    print("\nEvaluation du modèle Logistic Regression (baseline) :")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train_preprocessed, y_train)
    y_pred_lr = clf.predict(X_test_preprocessed)

    f1_baseline = f1_score(y_test, y_pred_lr)
    print(f"Baseline Logistic Regression F1 : {f1_baseline:.4f}")

    matrix(y_test, y_pred_lr)
    plot_loss_acc(history)

    # Sauvegarde du modèle et hyperparamètres (optionnel)
    if USE_TUNER:
        model.save("best_tuned_model.keras")
        with open("best_hyperparams.json", "w") as f:
            json.dump(best_hps.values, f, indent=4)

if __name__ == "__main__":
    main()
