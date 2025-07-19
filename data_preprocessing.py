import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import random
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import cv2

# Emotionen-Klassen entsprechend FER2013
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def augment_image(image_array):
    """
    Erweiterte Datenaugmentierung auf einem Graustufenbild (48, 48).
    """
    # In 3D-Array umwandeln (H, W, 1)
    if image_array.ndim == 2:
        image_array = image_array[..., np.newaxis]

    # Zufällige horizontale Spiegelung
    if random.random() > 0.5:
        image_array = np.fliplr(image_array)

    # Zufällige Rotation (-15 bis +15 Grad) mit OpenCV
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((24, 24), angle, 1)
        image_array = cv2.warpAffine(image_array, M, (48, 48), borderMode=cv2.BORDER_REFLECT)
        image_array = image_array[..., np.newaxis]

    # Zufällige Helligkeitsänderung (TensorFlow)
    if random.random() > 0.5:
        image_array = tf.image.adjust_brightness(image_array, random.uniform(-0.2, 0.2)).numpy()

    # Zufälliger Kontrast (TensorFlow)
    if random.random() > 0.5:
        image_array = tf.image.adjust_contrast(image_array, random.uniform(0.8, 1.2)).numpy()

    # Zufälliger Zoom (OpenCV)
    if random.random() > 0.5:
        zoom = random.uniform(0.9, 1.1)
        h, w = image_array.shape[:2]
        new_h, new_w = int(h * zoom), int(w * zoom)
        img_zoomed = cv2.resize(image_array, (new_w, new_h))
        if zoom < 1:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img_zoomed = cv2.copyMakeBorder(img_zoomed, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)
        else:
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            img_zoomed = img_zoomed[crop_h:crop_h + h, crop_w:crop_w + w]
        image_array = img_zoomed
        if image_array.ndim == 2:
            image_array = image_array[..., np.newaxis]

    # Zufällige Verschiebung (Translation, OpenCV)
    if random.random() > 0.5:
        max_shift = 4
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(image_array, M, (48, 48), borderMode=cv2.BORDER_REFLECT)
        image_array = shifted[..., np.newaxis]

    # Zufälliges Rauschen
    if random.random() > 0.5:
        noise = np.random.normal(0, 0.01, image_array.shape)
        image_array = np.clip(image_array + noise, 0, 1)

    # Zurück zu 2D für weitere Verarbeitung
    image_array = image_array.squeeze()
    return image_array

def load_and_preprocess_image(image_path, target_size=(48, 48), augment=False):
    """
    Lädt ein Bild und führt die erweiterte Vorverarbeitung durch.
    """
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img)
    scaler = StandardScaler()
    img_array = scaler.fit_transform(img_array.reshape(-1, 1)).reshape(img_array.shape)
    if augment:
        img_array = augment_image(img_array)
    return img_array

def prepare_dataset(data_dir, target_size=(48, 48), augment_training=True):
    images = []
    labels = []
    for emotion_idx, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(data_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue
        for img_name in os.listdir(emotion_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(emotion_dir, img_name)
                try:
                    should_augment = augment_training and 'train' in data_dir
                    img_array = load_and_preprocess_image(img_path, target_size, augment=should_augment)
                    images.append(img_array)
                    labels.append(emotion)
                except Exception as e:
                    print(f"Fehler beim Verarbeiten von {img_path}: {e}")
    X = np.array(images)
    y = np.array(labels)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    if 'train' in data_dir and len(X) > 0:
        X_reshaped = X.reshape(X.shape[0], -1)
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_reshaped, y)
        X = X_balanced.reshape(-1, target_size[0], target_size[1])
        y = y_balanced
    return X, y

def main():
    print("Verarbeite Trainingsdaten...")
    X_train, y_train = prepare_dataset('data/train', augment_training=True)
    print("Verarbeite Testdaten...")
    X_test, y_test = prepare_dataset('data/test', augment_training=False)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_test.npy', y_test)
    print(f"Trainingsdaten Form: {X_train.shape}")
    print(f"Testdaten Form: {X_test.shape}")
    print("Vorverarbeitung abgeschlossen!")

if __name__ == "__main__":
    main() 