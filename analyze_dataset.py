import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_dataset, EMOTIONS
import os
from PIL import Image
import cv2

def analyze_class_distribution(X_train, y_train, X_test, y_test):
    """Analysiert die Verteilung der Klassen im Trainings- und Testset."""
    plt.figure(figsize=(15, 5))
    
    # Trainingsdaten
    plt.subplot(1, 2, 1)
    train_counts = np.bincount(y_train)
    sns.barplot(x=EMOTIONS, y=train_counts)
    plt.title('Klassenverteilung im Trainingsset')
    plt.xticks(rotation=45)
    plt.ylabel('Anzahl der Bilder')
    
    # Testdaten
    plt.subplot(1, 2, 2)
    test_counts = np.bincount(y_test)
    sns.barplot(x=EMOTIONS, y=test_counts)
    plt.title('Klassenverteilung im Testset')
    plt.xticks(rotation=45)
    plt.ylabel('Anzahl der Bilder')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()
    
    return train_counts, test_counts

def analyze_image_quality(data_dir):
    """Analysiert die Bildqualität im Datensatz."""
    quality_metrics = {
        'size': [],
        'brightness': [],
        'contrast': [],
        'blur': []
    }
    
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        for img_name in os.listdir(emotion_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(emotion_dir, img_name)
                try:
                    # Bild laden
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    # Metriken berechnen
                    quality_metrics['size'].append(img.size)
                    quality_metrics['brightness'].append(np.mean(img))
                    quality_metrics['contrast'].append(np.std(img))
                    quality_metrics['blur'].append(cv2.Laplacian(img, cv2.CV_64F).var())
                except Exception as e:
                    print(f"Fehler bei {img_path}: {e}")
    
    # Visualisierung
    plt.figure(figsize=(15, 10))
    
    # Helligkeit
    plt.subplot(2, 2, 1)
    sns.histplot(quality_metrics['brightness'], bins=50)
    plt.title('Helligkeitsverteilung')
    plt.xlabel('Durchschnittliche Helligkeit')
    
    # Kontrast
    plt.subplot(2, 2, 2)
    sns.histplot(quality_metrics['contrast'], bins=50)
    plt.title('Kontrastverteilung')
    plt.xlabel('Standardabweichung')
    
    # Unschärfe
    plt.subplot(2, 2, 3)
    sns.histplot(quality_metrics['blur'], bins=50)
    plt.title('Unschärfe-Verteilung')
    plt.xlabel('Laplace-Varianz')
    
    # Bildgrößen
    plt.subplot(2, 2, 4)
    sns.histplot(quality_metrics['size'], bins=50)
    plt.title('Bildgrößen-Verteilung')
    plt.xlabel('Anzahl Pixel')
    
    plt.tight_layout()
    plt.savefig('image_quality_analysis.png')
    plt.close()
    
    return quality_metrics

def main():
    print("Lade Daten...")
    X_train, y_train = prepare_dataset('data/train', augment_training=False)
    X_test, y_test = prepare_dataset('data/test', augment_training=False)
    
    print("\nAnalysiere Klassenverteilung...")
    train_counts, test_counts = analyze_class_distribution(X_train, y_train, X_test, y_test)
    
    print("\nKlassenverteilung im Trainingsset:")
    for emotion, count in zip(EMOTIONS, train_counts):
        print(f"{emotion}: {count} Bilder")
    
    print("\nKlassenverteilung im Testset:")
    for emotion, count in zip(EMOTIONS, test_counts):
        print(f"{emotion}: {count} Bilder")
    
    print("\nAnalysiere Bildqualität...")
    quality_metrics = analyze_image_quality('data/train')
    
    # Zusammenfassung der Bildqualität
    print("\nBildqualität Zusammenfassung:")
    print(f"Durchschnittliche Helligkeit: {np.mean(quality_metrics['brightness']):.2f}")
    print(f"Durchschnittlicher Kontrast: {np.mean(quality_metrics['contrast']):.2f}")
    print(f"Durchschnittliche Unschärfe: {np.mean(quality_metrics['blur']):.2f}")
    print(f"Durchschnittliche Bildgröße: {np.mean(quality_metrics['size']):.2f} Pixel")

if __name__ == "__main__":
    main() 