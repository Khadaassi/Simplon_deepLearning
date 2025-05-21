# Telco Customer Churn Prediction - Deep Learning Project

## Contexte

Ce projet a été réalisé dans le cadre de la formation Simplon IA / Data, en binôme, pour répondre au besoin du client fictif **TelcoNova**, un opérateur télécom souhaitant **anticiper les départs de ses abonnés** (churn) afin d’optimiser ses campagnes de rétention.

Le client a fourni un extrait anonymisé de sa base CRM (dataset "Telco Customer Churn") pour entraîner un premier **prototype de modèle prédictif**, exploitable en production, avec un objectif clair : **livrer un modèle performant, traçable, reproductible et facilement intégrable** par les équipes MLOps.

---

## Objectif

Créer un pipeline de bout en bout avec :

* Préparation des données
* Modélisation via un MLP sous TensorFlow/Keras
* Gestion du déséquilibre de classes
* Suivi de l'entraînement via TensorBoard et callbacks
* Évaluation des performances
* Comparaison avec un modèle baseline (régression logistique)
* Export du modèle et des artefacts nécessaires à l’inférence

---

## Stack technique

* Python 3.12
* TensorFlow / Keras
* Scikit-learn
* Pandas / NumPy
* Matplotlib / Seaborn
* TensorBoard
* Git / GitHub

---

## Structure du projet

```
Simplon_deepLearning/
├── data/                        # Données brutes
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebook/                   # Analyse exploratoire
│   └── data_analysis.ipynb
├── utils/                      # Modules utilitaires : data prep, métriques, visualisation
│   ├── preprocessor.py
│   ├── metrics.py
│   ├── load_data.py
│   ├── build_pipeline.py
│   └── viz.py
├── main.py                    # Script principal : data prep, entraînement, évaluation
├── inference.py               # Script d'inférence sur nouvelles données
├── model/                     # Modèle MLP et recherche d'hyperparamètres
│   └── build_model.py
├── tuner_results/             # Résultats KerasTuner (RandomSearch)
│   └── churn_tuning/
├── results/                   # Graphiques générés automatiquement
│   └── plots/
├── logs/                      # Logs TensorBoard pour le suivi de l'entraînement
├── best_model.keras           # Modèle entraîné sauvegardé
├── best_tuned_model.keras     # Modèle issu du tuning
├── best_hyperparams.json      # Fichier JSON des meilleurs hyperparamètres
├── requirements.txt           # Dépendances Python
└── README.md
```

---

## Modélisation

Le module `build_model.py` contient la logique de création du modèle MLP ainsi que la recherche d'hyperparamètres avec KerasTuner. Il inclut :

* `build_model()` : construction d’un réseau de neurones simple avec deux couches cachées (128 et 64 neurones, ReLU) et une sortie softmax.
* `get_class_weights()` : calcul des poids de classe pour la gestion du déséquilibre.
* `MyHyperModel` : classe personnalisée pour la recherche d’architecture via KerasTuner (nombre de couches, taille des couches, learning rate).
* `run_hyperparameter_search()` : exécution d’un `RandomSearch` avec validation croisée.

### Modèle MLP (TensorFlow / Keras)

* Architecture pensée pour les données tabulaires
* ≥ 2 couches cachées avec `ReLU`
* Fonction de perte : `binary_crossentropy`
* Optimiseur : `Adam`
* Gestion du déséquilibre avec `class_weight`
* Callbacks : `EarlyStopping`, `ModelCheckpoint`, `TensorBoard`
* Recherche d'hyperparamètres avec KerasTuner (option activable)

### Baseline

Un modèle de régression logistique est utilisé comme référence pour mesurer le gain apporté par le réseau de neurones.

---

## Évaluation

### Métriques suivies :

* F1-score, ROC-AUC, Recall
* Matrice de confusion
* Courbes de loss & accuracy (avec validation)
* Comparaison avec baseline

### Résultats obtenus

Le modèle MLP atteint les performances suivantes sur le jeu de test (1409 exemples) :

| Classe          | Précision | Rappel | F1-score | Support |
| --------------- | --------- | ------ | -------- | ------- |
| Yes             | 0.51      | 0.77   | 0.61     | 374     |
| No              | 0.90      | 0.73   | 0.80     | 1035    |
| Accuracy        |           |        | 0.74     |         |
| ROC-AUC         |           |        | 0.8228   |         |
| F1-score global |           |        | 0.8047   |         |
| Macro avg       | 0.70      | 0.75   | 0.71     | 1409    |
| Weighted avg    | 0.79      | 0.74   | 0.75     | 1409    |

Critère du projet atteint : ROC-AUC > 0.80 et F1-score global > 0.60

---

## Inference

Un script `inference.py` permet de :

* Charger un modèle sauvegardé (`.keras`)
* Recharger le pipeline de preprocessing complet (`preprocessor.joblib`)
* Prédire le churn sur de nouvelles données (format `.csv`)
* Générer des colonnes `Churn_Prob` (probabilité) et `Churn_Pred` (classe prédite)

### Exemple d'utilisation :

```bash
python inference.py --input data/new_customers.csv
```

### Exemple de sortie :

```
Résultats de la prédiction :

╒══════════════╤══════════════╤══════════════════╕
│ customerID   │ Churn_Prob   │ Réponse          │
╞══════════════╪══════════════╪══════════════════╡
│ 7795-CFOCW   │ 0.9847       │ Reste fidèle     │
│ 9237-HQITU   │ 0.0918       │ Va quitter       │
│ 9305-CDSKC   │ 0.0367       │ Va quitter       │
│ 1452-KIOVK   │ 0.3745       │ Va quitter       │
│ 6713-OKOMC   │ 0.6581       │ Reste fidèle     │
│ 7892-POOKP   │ 0.1502       │ Va quitter       │
╘══════════════╧══════════════╧══════════════════╛
```

> Interprétation : une probabilité inférieure à 0.5 signifie un risque de départ (churn). Le modèle restitue une réponse lisible : "Va quitter" ou "Reste fidèle".

## Reproductibilité

* Seeds fixés pour garantir la reproductibilité
* Callbacks utilisés : `EarlyStopping`, `ModelCheckpoint`, `TensorBoard`
* Possibilité d’enregistrer le meilleur modèle et ses hyperparamètres (`best_model.keras`, `best_hyperparams.json`)

---

## Collaboration Git

* Travail en binôme via GitHub
* 1 branche = 1 feature
* Pull requests systématiques avec descriptions et validations croisées

---

## Recommandations d’utilisation

### Lancer l'entraînement :

```bash
python main.py
```

### Lancer une prédiction :

```bash
python inference.py --input data/new_customers.csv
```

### Afficher les logs TensorBoard :

```bash
tensorboard --logdir=results/logs
```

---

## Données

* Dataset : `Telco Customer Churn`
* Format CSV
* Pré-nettoyé en grande partie (mais vérification & transformation nécessaires)
* Pas de données personnelles (RGPD respecté)

---

## Auteurs

Projet réalisé par :

* **Khadija Aassi**: [GitHub](https://github.com/Khadaassi)
* **Wael Bensoltana**: [GitHub](https://github.com/wbensolt)

---

## Liens utiles

* [Projet GitHub](https://github.com/Khadaassi/Simplon_deepLearning)
* [TensorBoard](http://localhost:6006) (lorsqu’activé)
