import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Dossier de sortie
OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # crée le dossier s'il n'existe pas

def matrix(y_test, y_pred, filename="confusion_matrix.png"):
    print("Matrice de Confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)   

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matrice de Confusion")
    plt.xlabel("Classe Prédite")
    plt.ylabel("Classe Réelle")

    # Sauvegarde
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    print(f"📁 Confusion matrix enregistrée dans : {save_path}")

    plt.show(block=False)
    plt.pause(3)
    plt.close()

def plot_loss_acc(history, validation=True, filename="training_curves.png"):
    """
    Trace la loss et l'accuracy du modèle pendant l'entraînement et enregistre le graph.
    """
    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    if validation and 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Évolution de la Loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    if validation and 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Évolution de l'Accuracy")
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Sauvegarde
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    print(f"📁 Graphique entraînement enregistré dans : {save_path}")

    plt.show(block=False)
    plt.pause(5)
    plt.close()
