import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model_paths = {
    'Custom CNN': '../emotion_model.h5',
    'VGGFace': 'vggface_emotion_model.h5',
    'ResNet50': 'resnet50_emotion_model.h5'
}

# Testdaten laden
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

results = {}
for name, path in model_paths.items():
    try:
        model = tf.keras.models.load_model(path)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_classes)
        results[name] = {
            'accuracy': test_acc,
            'loss': test_loss,
            'report': report,
            'cm': cm
        }
        print(f"\n=== {name} ===")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Loss: {test_loss:.4f}")
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
    except Exception as e:
        print(f'Fehler bei {name}:', e)

# Ergebnisse tabellarisch anzeigen
metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1-score']
table = []
for name, res in results.items():
    row = [name, res['accuracy'], res['loss'],
           res['report']['weighted avg']['precision'],
           res['report']['weighted avg']['recall'],
           res['report']['weighted avg']['f1-score']]
    table.append(row)
df = pd.DataFrame(table, columns=['Modell', 'Accuracy', 'Loss', 'Precision', 'Recall', 'F1-Score'])
print("\nVergleichstabelle:")
print(df)

def plot_confusion_matrices(results, class_names):
    for name, res in results.items():
        plt.figure(figsize=(8,6))
        sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()

plot_confusion_matrices(results, class_names)
