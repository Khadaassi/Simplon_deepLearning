import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from utils.load_data import load_data
from utils.preprocessor import preprocess_data
from utils.build_pipeline import build_pipeline
from utils.metrics  import metrics
from utils.build_model  import build_model
import sys

data = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def main():

    original_data = preprocess_data(data)
    #print(original_data.shape)
    [X_train_preprocessed, X_val_preprocessed, X_test_preprocessed, y_train, y_val, y_test] = build_pipeline(original_data)

    # Encodage des labels en one-hot
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, 2)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  2)
    y_train_cat[:5], y_val_cat[:5], y_test_cat[:5]

    model = build_model(X_train_preprocessed, 2 )

    history = model.fit(
        X_train_preprocessed, y_train_cat,
        validation_data=(X_val_preprocessed, y_val_cat),
        epochs=20,
        batch_size=16,
        verbose=1
    )

    metrics(model, X_test_preprocessed, y_test, y_test_cat)




if __name__ == "__main__":
    main()
