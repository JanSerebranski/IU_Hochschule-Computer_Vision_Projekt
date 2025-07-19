import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_dataset

def evaluate_model(model_path, test_data_path):
    """
    Evaluiert das trainierte Modell auf dem Testdatensatz.
    
    Args:
        model_path: Pfad zum trainierten Modell
        test_data_path: Pfad zum Testdatensatz
    """
    # Modell laden
    model = tf.keras.models.load_model(model_path)
    
    # Testdaten laden
    X_test, y_test = prepare_dataset(test_data_path, augment_training=False)
    
    # Vorhersagen machen
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # y_test kann entweder One-Hot oder Label-Array sein
    if len(y_test.shape) == 1:
        y_true_classes = y_test
    else:
        y_true_classes = np.argmax(y_test, axis=1)
    
    # Metriken berechnen
    test_loss, test_accuracy = model.evaluate(X_test, y_true_classes, verbose=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Klassennamen
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    # Ergebnisse ausgeben
    print("\nModell-Evaluierung:")
    print(f"Test Genauigkeit: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\nKlassifikationsbericht:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # Confusion Matrix visualisieren
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Fehleranalyse
    print("\nFehleranalyse:")
    for i, class_name in enumerate(class_names):
        # Falsch klassifizierte Bilder pro Klasse
        false_positives = np.sum((y_pred_classes == i) & (y_true_classes != i))
        false_negatives = np.sum((y_pred_classes != i) & (y_true_classes == i))
        
        print(f"\n{class_name}:")
        print(f"Falsch positive: {false_positives}")
        print(f"Falsch negative: {false_negatives}")
        
        # Häufigste Fehlklassifikationen
        if false_positives > 0:
            print("Häufigste Fehlklassifikationen als", class_name)
            for j, other_class in enumerate(class_names):
                if i != j:
                    count = np.sum((y_pred_classes == i) & (y_true_classes == j))
                    if count > 0:
                        print(f"  - Als {other_class} klassifiziert: {count}")

if __name__ == "__main__":
    evaluate_model('emotion_model.h5', 'data/test') 