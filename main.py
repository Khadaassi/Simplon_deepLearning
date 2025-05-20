import datetime
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf
from utils.load_data import load_data
from utils.preprocessor import preprocess_data
from utils.build_pipeline import build_pipeline
from utils.metrics  import metrics
from utils.build_model  import build_model, get_class_weights
import sys
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

data = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():

    set_seed(42)
    original_data = preprocess_data(data)
    #print(original_data.shape)
    [X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test] = build_pipeline(original_data)

    # Encodage des labels en one-hot
    print("XXXX",y_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  2)
    y_train_cat[:5], y_val_cat[:5], y_test_cat[:5]

    model = build_model(X_train_preprocessed, 2 )

    class_weights = get_class_weights(y_train)
    

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_cb = ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor='val_loss', mode='min'
    )

    print(class_weights)
    history = model.fit(
        X_train_preprocessed, y_train_cat,
        validation_data=(X_val_preprocessed, y_val_cat),
        epochs=20, #class_weight=class_weights,
        batch_size=16,
        verbose=1,
        callbacks=[tensorboard_cb, checkpoint_cb]
    )

    metrics(model, X_test_preprocessed, y_test, y_test_cat)

    clf = LogisticRegression(class_weight='balanced')
    clf.fit(X_train_preprocessed, y_train)
    y_pred = clf.predict(X_test_preprocessed)

    f1_baseline = f1_score(y_test, y_pred)
    print(f"Baseline Logistic Regression F1 : {f1_baseline:.4f}")


if __name__ == "__main__":
    main()
