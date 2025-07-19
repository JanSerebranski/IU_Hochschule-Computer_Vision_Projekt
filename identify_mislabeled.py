import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_dataset, EMOTIONS
import os
from PIL import Image
import cv2
import json

def load_trained_model(model_path='best_model.h5'):
    """Lädt das trainierte Modell."""
    try:
        return load_model(model_path)
    except:
        print("Kein vortrainiertes Modell gefunden. Verwende zufällige Vorhersagen für Demonstration.")
        return None

def predict_emotions(model, images, threshold=0.7):
    """Macht Vorhersagen für die Bilder und gibt Unsicherheiten zurück."""
    if model is None:
        # Simuliere zufällige Vorhersagen für Demonstration
        return np.random.random((len(images), len(EMOTIONS)))
    
    predictions = model.predict(images[..., np.newaxis])
    return predictions

def find_potential_mislabeled(model, X, y, threshold=0.7):
    """Findet potenziell falsch gelabelte Bilder."""
    predictions = predict_emotions(model, X)
    
    # Berechne Unsicherheit für jedes Bild
    uncertainties = []
    for i, (pred, true_label) in enumerate(zip(predictions, y)):
        # Unsicherheit ist die Differenz zwischen der höchsten Vorhersage und der Vorhersage für die wahre Klasse
        max_pred = np.max(pred)
        true_pred = pred[true_label]
        uncertainty = max_pred - true_pred
        
        # Wenn die Unsicherheit hoch ist oder die Vorhersage für die wahre Klasse niedrig ist
        if uncertainty > threshold or true_pred < 0.3:
            uncertainties.append({
                'index': int(i),
                'uncertainty': float(uncertainty),
                'true_label': int(true_label),
                'predicted_label': int(np.argmax(pred)),
                'true_probability': float(true_pred),
                'predicted_probability': float(max_pred)
            })
    
    return sorted(uncertainties, key=lambda x: x['uncertainty'], reverse=True)

def visualize_potential_mislabeled(X, y, uncertainties, n_samples=10, filename='potential_mislabeled.png'):
    """Visualisiert potenziell falsch gelabelte Bilder."""
    plt.figure(figsize=(15, 2*n_samples))
    
    for i, item in enumerate(uncertainties[:n_samples]):
        plt.subplot(n_samples, 1, i+1)
        plt.imshow(X[item['index']], cmap='gray')
        plt.title(f'Wahre Klasse: {EMOTIONS[item["true_label"]]}, '
                 f'Vorhergesagt: {EMOTIONS[item["predicted_label"]]}, '
                 f'Unsicherheit: {item["uncertainty"]:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_label_consistency(X, y, n_neighbors=5):
    """Analysiert die Konsistenz der Labels mit ähnlichen Bildern."""
    from sklearn.neighbors import NearestNeighbors
    
    # Flache die Bilder für den Vergleich
    X_flat = X.reshape(X.shape[0], -1)
    
    # Finde ähnliche Bilder
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_flat)
    distances, indices = nbrs.kneighbors(X_flat)
    
    # Analysiere Label-Konsistenz
    inconsistencies = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        # Überspringe das erste Element (das Bild selbst)
        neighbor_labels = y[idx[1:]]
        true_label = y[i]
        
        # Berechne Konsistenz
        consistency = np.mean(neighbor_labels == true_label)
        if consistency < 0.5:  # Weniger als die Hälfte der Nachbarn hat das gleiche Label
            inconsistencies.append({
                'index': int(i),
                'consistency': float(consistency),
                'true_label': int(true_label),
                'neighbor_labels': neighbor_labels.tolist(),
                'distances': dist[1:].tolist()  # Distanzen zu den Nachbarn
            })
    
    return sorted(inconsistencies, key=lambda x: x['consistency'])

def analyze_class_distribution(mislabeled_data, y, dataset_name):
    """Analysiert die Verteilung der falsch gelabelten Bilder über die Klassen."""
    class_counts = {emotion: 0 for emotion in EMOTIONS}
    for item in mislabeled_data:
        class_counts[EMOTIONS[item['true_label']]] += 1
    
    # Erstelle Visualisierung
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(f'Verteilung falsch gelabelter Bilder - {dataset_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'mislabeled_distribution_{dataset_name}.png')
    plt.close()
    
    return class_counts

def main():
    print("Lade Daten...")
    X_train, y_train = prepare_dataset('data/train', augment_training=False)
    X_test, y_test = prepare_dataset('data/test', augment_training=False)
    
    print("\nLade Modell...")
    model = load_trained_model()
    
    print("\nSuche nach potenziell falsch gelabelten Bildern...")
    # Für Trainingsdaten
    train_mislabeled = find_potential_mislabeled(model, X_train, y_train)
    print(f"\nGefundene potenziell falsch gelabelte Trainingsbilder: {len(train_mislabeled)}")
    
    # Für Testdaten
    test_mislabeled = find_potential_mislabeled(model, X_test, y_test)
    print(f"Gefundene potenziell falsch gelabelte Testbilder: {len(test_mislabeled)}")
    
    print("\nAnalysiere Label-Konsistenz...")
    train_inconsistencies = analyze_label_consistency(X_train, y_train)
    print(f"\nGefundene inkonsistente Trainingslabels: {len(train_inconsistencies)}")
    
    test_inconsistencies = analyze_label_consistency(X_test, y_test)
    print(f"Gefundene inkonsistente Testlabels: {len(test_inconsistencies)}")
    
    print("\nAnalysiere Klassenverteilung...")
    train_distribution = analyze_class_distribution(train_mislabeled, y_train, 'train')
    test_distribution = analyze_class_distribution(test_mislabeled, y_test, 'test')
    
    print("\nVisualisiere potenziell falsch gelabelte Bilder...")
    visualize_potential_mislabeled(X_train, y_train, train_mislabeled, filename='potential_mislabeled_train.png')
    visualize_potential_mislabeled(X_test, y_test, test_mislabeled, filename='potential_mislabeled_test.png')
    
    # Speichere die Ergebnisse als JSON
    results = {
        'train_mislabeled': train_mislabeled,
        'test_mislabeled': test_mislabeled,
        'train_inconsistencies': train_inconsistencies,
        'test_inconsistencies': test_inconsistencies,
        'train_distribution': train_distribution,
        'test_distribution': test_distribution
    }
    
    with open('mislabeled_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main() 