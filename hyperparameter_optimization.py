import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from emotion_model import create_emotion_cnn
import tensorflow as tf

def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Führt die Hyperparameter-Optimierung durch.
    """
    # Modell für GridSearch vorbereiten
    model = KerasClassifier(
        build_fn=lambda: create_emotion_cnn(),
        verbose=0
    )

    # Parameter-Grid definieren
    param_grid = {
        'batch_size': [16, 32, 64],
        'epochs': [30, 50],
        'optimizer__learning_rate': [0.0001, 0.001, 0.01],
        'optimizer__beta_1': [0.9, 0.95],
        'optimizer__beta_2': [0.999, 0.9999]
    }

    # GridSearchCV initialisieren
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Optimierung durchführen
    print("Starte Hyperparameter-Optimierung...")
    grid_result = grid.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # Ergebnisse ausgeben
    print("\nBeste Parameter gefunden:")
    print(grid_result.best_params_)
    print("\nBeste Validierungsgenauigkeit:")
    print(grid_result.best_score_)

    return grid_result.best_params_

def main():
    # Daten laden
    print("Lade verarbeitete Daten...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')

    # Validierungsdaten aus Trainingsdaten erstellen (20%)
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]

    # Datenform für CNN vorbereiten
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

    # Hyperparameter-Optimierung durchführen
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val)

    # Beste Parameter speichern
    np.save('best_hyperparameters.npy', best_params)
    print("\nOptimierte Hyperparameter wurden gespeichert.")

if __name__ == "__main__":
    main() 