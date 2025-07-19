import numpy as np
import tensorflow as tf
from emotion_model import create_emotion_cnn, evaluate_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def test_model_architecture():
    print("1. Teste Modellarchitektur...")
    
    # Modell erstellen
    model = create_emotion_cnn()
    
    # Überprüfe Modellstruktur
    print("\nModellstruktur:")
    model.summary()
    
    # Teste Forward-Pass
    print("\nTeste Forward-Pass...")
    test_input = np.random.rand(1, 48, 48, 1)
    test_output = model.predict(test_input)
    
    print(f"✓ Eingabedimensionen: {test_input.shape}")
    print(f"✓ Ausgabedimensionen: {test_output.shape}")
    print(f"✓ Ausgabewertebereich: [{test_output.min():.4f}, {test_output.max():.4f}]")
    print(f"✓ Summe der Ausgabewahrscheinlichkeiten: {np.sum(test_output):.4f}")

def test_model_predictions():
    print("\n2. Teste Modellvorhersagen...")
    
    # Lade das trainierte Modell
    try:
        model = tf.keras.models.load_model('emotion_model.h5')
        print("✓ Trainiertes Modell erfolgreich geladen")
    except:
        print("✗ Kein trainiertes Modell gefunden. Erstelle neues Modell für Tests...")
        model = create_emotion_cnn()
    
    # Lade Testdaten
    try:
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
        print(f"✓ Testdaten geladen: {X_test.shape}")
    except:
        print("✗ Keine Testdaten gefunden. Erstelle Beispieldaten...")
        X_test = np.random.rand(10, 48, 48, 1)
        y_test = np.random.randint(0, 7, 10)
    
    # Teste Einzelvorhersagen
    print("\nTeste Einzelvorhersagen...")
    single_pred = model.predict(X_test[0:1])
    print(f"✓ Einzelvorhersage Form: {single_pred.shape}")
    print(f"✓ Vorhergesagte Emotion: {np.argmax(single_pred)}")
    print(f"✓ Wahrscheinlichkeiten: {single_pred[0]}")
    
    # Teste Batch-Vorhersagen
    print("\nTeste Batch-Vorhersagen...")
    batch_pred = model.predict(X_test[:5])
    print(f"✓ Batch-Vorhersage Form: {batch_pred.shape}")
    print(f"✓ Vorhergesagte Emotionen: {np.argmax(batch_pred, axis=1)}")
    
    # Evaluierung
    print("\nEvaluierung des Modells...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"✓ Test Genauigkeit: {test_accuracy:.4f}")
    print(f"✓ Test Verlust: {test_loss:.4f}")
    
    # Detaillierte Metriken
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nKlassifikationsbericht:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("\n✓ Confusion Matrix gespeichert als 'confusion_matrix.png'")

if __name__ == "__main__":
    test_model_architecture()
    test_model_predictions() 