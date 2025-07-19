from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import requests
import cv2
import logging
from reddit_api import RedditAPI
from reddit_config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
from emotion_model import SqueezeExcitation

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Globale Variablen
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = None
face_cascade = None
reddit_api = RedditAPI()

def load_models():
    """Lädt das trainierte Modell und den Gesichtserkennungs-Cascade."""
    global model, face_cascade
    if model is None:
        try:
            model = tf.keras.models.load_model('best_model.h5', custom_objects={'SqueezeExcitation': SqueezeExcitation})
            logger.info("Emotion-Modell erfolgreich geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden des Emotion-Modells: {e}")
            raise
    
    if face_cascade is None:
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("Gesichtserkennungs-Cascade erfolgreich geladen")
        except Exception as e:
            logger.error(f"Fehler beim Laden des Gesichtserkennungs-Cascade: {e}")
            raise

def preprocess_image(image):
    """Bereitet ein Bild für die Vorhersage vor."""
    try:
        # Bild in Graustufen konvertieren, aber nur wenn nötig
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                gray = image
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.convert('L')
            gray = np.array(gray)
        # Histogrammausgleich
        gray = cv2.equalizeHist(gray)
        # Z-Score Normalisierung
        mean = np.mean(gray)
        std = np.std(gray)
        gray = (gray - mean) / (std + 1e-7)
        # Dimension für Batch und Kanal hinzufügen
        gray = gray.reshape(1, 48, 48, 1)
        return gray
    except Exception as e:
        logger.error(f"Fehler bei der Bildvorverarbeitung: {e}")
        raise

def detect_faces(image):
    """Erkennt Gesichter in einem Bild und speichert Debug-Bild mit Rechtecken."""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Prüfe, ob das Bild bereits Graustufen ist
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            gray = image
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(24, 24)
        )
        logger.info(f"Anzahl erkannter Gesichter: {len(faces)}")
        # Debug-Bild mit Rechtecken speichern
        if len(faces) > 0:
            debug_dir = 'debug_faces'
            os.makedirs(debug_dir, exist_ok=True)
            debug_img = image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            debug_path = os.path.join(debug_dir, f'debug_{np.random.randint(100000)}.jpg')
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"Debug-Bild gespeichert: {debug_path}")
        return faces
    except Exception as e:
        logger.error(f"Fehler bei der Gesichtserkennung: {e}")
        return []

def analyze_faces(image, faces):
    """Analysiert die Emotionen in erkannten Gesichtern."""
    results = []
    for (x, y, w, h) in faces:
        try:
            # Gesicht extrahieren und auf 48x48 skalieren
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            
            # Vorverarbeitung und Vorhersage
            processed_face = preprocess_image(face)
            predictions = model.predict(processed_face)[0]
            
            # Emotion mit höchster Wahrscheinlichkeit
            emotion_idx = np.argmax(predictions)
            emotion = EMOTIONS[emotion_idx]
            confidence = float(predictions[emotion_idx])
            
            # Alle Emotionen mit Wahrscheinlichkeiten
            emotion_probs = {
                EMOTIONS[i]: float(pred) 
                for i, pred in enumerate(predictions)
            }
            
            results.append({
                'position': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': emotion_probs
            })
        except Exception as e:
            logger.error(f"Fehler bei der Analyse eines Gesichts: {e}")
            continue
    
    return results

@app.route('/')
def home():
    """Rendert die Hauptseite."""
    return render_template('index.html')

@app.route('/reddit/feed')
def reddit_feed():
    """Zeigt den Reddit-Feed an (nur Bild-Posts)."""
    media = reddit_api.get_image_posts()
    return render_template('reddit_feed.html', media=media)

@app.route('/analyze/reddit/<post_id>')
def analyze_reddit_post(post_id):
    """Analysiert ein Reddit-Bild-Post."""
    media = reddit_api.get_post_details(post_id)
    if not media or not media.get('image_url'):
        return jsonify({'success': False, 'error': 'Ungültiger Post oder kein Bild vorhanden'})
    response = requests.get(media['image_url'])
    image = Image.open(io.BytesIO(response.content))
    image = np.array(image)
    faces = detect_faces(image)
    if len(faces) == 0:
        return jsonify({'success': False, 'error': 'Keine Gesichter erkannt'})
    results = analyze_faces(image, faces)
    return jsonify({'success': True, 'faces': results, 'media': media})

@app.route('/analyze/reddit_url', methods=['POST'])
def analyze_reddit_url():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'success': False, 'error': 'Keine URL angegeben'}), 400
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({'success': False, 'error': 'Bild konnte nicht geladen werden (Statuscode)'}), 400
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            return jsonify({'success': False, 'error': 'Die URL verweist nicht direkt auf ein Bild. Bitte eine direkte Bild-URL verwenden.'}), 400
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        image_np = np.array(image)
        faces = detect_faces(image_np)
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'Kein Gesicht erkannt'})
        results = analyze_faces(image_np, faces)
        # Rückgabe wie bei batch-analyze
        return jsonify({'success': True, 'results': [{
            'filename': url,
            'faces': results
        }]})
    except Exception as e:
        logger.error(f"Fehler bei der Reddit-URL-Analyse: {e}")
        return jsonify({'success': False, 'error': 'Fehler beim Laden oder Verarbeiten des Bildes. Prüfe die URL.'})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analysiert ein hochgeladenes Bild."""
    try:
        # Bild aus dem Request extrahieren
        file = request.files['image']
        image = Image.open(file.stream)
        image = np.array(image)
        
        # Gesichter erkennen und analysieren
        faces = detect_faces(image)
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'error': 'Keine Gesichter erkannt'
            })
        
        results = analyze_faces(image, faces)
        
        return jsonify({
            'success': True,
            'faces': results
        })
        
    except Exception as e:
        logger.error(f"Fehler bei der Bildanalyse: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """Analysiert mehrere Bilder in einem Batch."""
    try:
        files = request.files.getlist('images')
        results = []
        
        for file in files:
            try:
                image = Image.open(file.stream)
                image = np.array(image)
                faces = detect_faces(image)
                if len(faces) > 0:
                    face_results = analyze_faces(image, faces)
                    results.append({
                        'filename': file.filename,
                        'faces': face_results
                    })
                else:
                    results.append({
                        'filename': file.filename,
                        'error': 'Kein Gesicht erkannt'
                    })
            except Exception as e:
                logger.error(f"Fehler bei der Analyse von {file.filename}: {e}")
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Fehler bei der Batch-Analyse: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_models()
    app.run(debug=True) 