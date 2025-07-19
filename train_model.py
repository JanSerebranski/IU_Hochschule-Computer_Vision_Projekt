import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
from emotion_model import create_emotion_cnn
from data_preprocessing import prepare_dataset

def plot_training_history(history):
    """Plottet die Trainingshistorie."""
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss Plot
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy Plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    # Daten laden
    print("Lade und verarbeite Daten...")
    X_train, y_train = prepare_dataset('data/train', augment_training=True)
    X_test, y_test = prepare_dataset('data/test', augment_training=False)
    
    # Modell erstellen
    print("Erstelle Modell...")
    model = create_emotion_cnn()
    
    # Callbacks definieren
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Modell kompilieren
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Training durchführen
    print("Starte Training...")
    history = model.fit(
        X_train[..., np.newaxis], y_train,
        batch_size=32,
        epochs=100,  # Maximale Anzahl von Epochen
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Historie speichern
    np.save('training_history.npy', history.history)
    
    # Training visualisieren
    plot_training_history(history)
    
    # Modell speichern
    model.save('emotion_model.h5')
    
    # Evaluation
    print("\nEvaluierung auf Testdaten:")
    test_loss, test_accuracy = model.evaluate(X_test[..., np.newaxis], y_test, verbose=1)
    print(f"Test Genauigkeit: {test_accuracy:.4f}")
    print(f"Test Verlust: {test_loss:.4f}")

if __name__ == "__main__":
    train_model() 