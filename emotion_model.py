import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.applications import EfficientNetB0

class SqueezeExcitation(layers.Layer):
    def __init__(self, ratio=16, *args, **kwargs):
        super(SqueezeExcitation, self).__init__(*args, **kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excitation = layers.Dense(input_shape[-1] // self.ratio, activation='relu')
        self.excitation2 = layers.Dense(input_shape[-1], activation='sigmoid')

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation(x)
        x = self.excitation2(x)
        return inputs * tf.reshape(x, [-1, 1, 1, x.shape[-1]])

def residual_block(x, filters, kernel_size=3):
    """
    Erstellt einen Residual Block mit Squeeze-and-Excitation.
    """
    shortcut = x
    
    # Erster Convolutional Layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Zweiter Convolutional Layer
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Squeeze-and-Excitation
    x = SqueezeExcitation()(x)
    
    # Shortcut Connection
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def attention_block(x, g, filters):
    """
    Erstellt einen Attention Block.
    """
    theta_x = layers.Conv2D(filters, 1)(x)
    phi_g = layers.Conv2D(filters, 1)(g)
    
    f = layers.Activation('relu')(layers.add([theta_x, phi_g]))
    f = layers.Conv2D(1, 1)(f)
    f = layers.Activation('sigmoid')(f)
    
    return layers.multiply([x, f])

def create_emotion_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Erstellt ein verbessertes CNN-Modell für Emotionserkennung.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual Blocks mit Attention
    x1 = residual_block(x, 64)
    x2 = residual_block(x1, 128)
    x3 = residual_block(x2, 256)
    
    # Attention Mechanism
    x2_att = attention_block(x2, x3, 128)
    x1_att = attention_block(x1, x2_att, 64)
    
    # Global Features
    x = layers.GlobalAveragePooling2D()(x3)
    
    # Fully Connected Layers mit Dropout
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_learning_rate_scheduler():
    """
    Erstellt einen Learning Rate Scheduler.
    """
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    
    return lr_schedule

def train_model(X_train, y_train, X_test, y_test, hyperparameters=None):
    """
    Trainiert das verbesserte Emotionserkennungsmodell.
    """
    # Modell erstellen
    model = create_emotion_cnn()
    
    # Standard-Hyperparameter
    batch_size = 32
    epochs = 50
    
    # Optimierte Hyperparameter verwenden, falls vorhanden
    if hyperparameters is not None:
        batch_size = hyperparameters.get('batch_size', batch_size)
        epochs = hyperparameters.get('epochs', epochs)
    
    # Learning Rate Scheduler
    lr_schedule = create_learning_rate_scheduler()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Modell kompilieren
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )
    
    # Modell trainieren
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on test data and prints metrics.
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Genauigkeit: {test_accuracy:.4f}")
    print(f"Test Verlust: {test_loss:.4f}")

def create_efficientnet_emotion(input_shape=(96, 96, 3), num_classes=7, trainable_base=False):
    """
    Erstellt ein EfficientNetB0-Modell mit vortrainierten ImageNet-Gewichten für Emotionserkennung.
    Die Basis kann eingefroren oder feingetunt werden.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = trainable_base  # Für Feintuning ggf. auf True setzen

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    # Daten laden
    print("Lade verarbeitete Daten...")
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Datenform für CNN vorbereiten
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Optimierte Hyperparameter laden, falls vorhanden
    try:
        hyperparameters = np.load('best_hyperparameters.npy', allow_pickle=True).item()
        print("Gefundene optimierte Hyperparameter werden verwendet.")
    except:
        hyperparameters = None
        print("Keine optimierten Hyperparameter gefunden. Verwende Standardwerte.")
    
    print("Starte Modelltraining...")
    model, history = train_model(X_train, y_train, X_test, y_test, hyperparameters)
    
    print("\nEvaluierung des Modells...")
    evaluate_model(model, X_test, y_test)
    
    # Modell speichern
    model.save('emotion_model.h5')
    print("\nModell wurde als 'emotion_model.h5' gespeichert.")

    from emotion_model import train_model
    train_model()  # oder entsprechendes Training-Skript 