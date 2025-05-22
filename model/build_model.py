import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt

def build_model(X_train, num_classes):
    """Génération du modèle

    Args:
        X_train : les données d'entraînement
        num_classes : le nombre de classes

    Returns:
        model : modèle du réseau de neurones
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def get_class_weights(y_train):
    """pondération de classes en cas d'un non équilibre détecté

    Args:
        y_train : les données d'entraînement

    Returns:
        class_weights: la pondération des classes 
    """
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    return {i: w for i, w in enumerate(class_weights)}

class MyHyperModel(kt.HyperModel):
    """Classe MyHyperModel permet de chercher un modèle de réseau de neurones performant
                    suivant la métrique définie 
    """
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes

    def build(self, hp):
        model = tf.keras.Sequential()

        #Couche d'entrée
        model.add(tf.keras.layers.Input(shape=(self.input_dim,)))

        #Couche cachée
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
                activation='relu'
            ))

        #Couche de sortie
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

def run_hyperparameter_search(X_train, y_train_cat, X_val, y_val_cat, num_classes):
    """run_hyperparameter_search permet de chercher les meilleurs paramètres

    Args:
        X_train : donnée d'entraînement 
        y_train_cat : donnée cible d'entraînement 
        X_val : donnée de validation
        y_val_cat : donnée cible de validation
        num_classes : nombre de classes

    Returns:
        _best_model, best_hps : meilleur modèle, meilleurs hyperparamètres
    """
    tuner = kt.RandomSearch(
        hypermodel=MyHyperModel(input_dim=X_train.shape[1], num_classes=num_classes),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='churn_tuning'
    )

    tuner.search(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=10,
        batch_size=32
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters()[0]

    return best_model, best_hps
