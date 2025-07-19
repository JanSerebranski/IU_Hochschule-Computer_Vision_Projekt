# IU_Hochschule-Computer_Vision_Projekt

Emotionserkennung in Gesichtern - Computer Vision Projekt


Projektbeschreibung
Dieses Projekt implementiert ein Computer-Vision-System zur automatisierten Emotionserkennung in Gesichtern. Es wurde im Rahmen eines Hochschulprojekts entwickelt und umfasst die Implementierung verschiedener Deep-Learning-Architekturen, eine Webanwendung zur Bildanalyse sowie die Integration von Social-Media-Feeds.


Hauptfunktionen
Emotionsklassifikation: Erkennung von 7 Basisemotionen (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
Webanwendung: Upload und Analyse von Bildern über eine Flask-basierte Web-Oberfläche
Social-Media-Integration: Automatische Analyse von Reddit-Bildern
Modellvergleich: Evaluation verschiedener CNN-Architekturen (Custom CNN, VGGFace)


Datensatz
Das System wurde auf dem FER-2013-Datensatz trainiert, der über 35.000 Gesichtsbilder mit 7 verschiedenen Emotionsklassen enthält. Die Bilder wurden auf 48x48 Pixel skaliert und als Graustufenbilder verarbeitet.


Technologie-Stack
Backend: Python, Flask, TensorFlow/Keras
Frontend: HTML, CSS, JavaScript
Deep Learning: Custom CNN mit Residual- und Attention-Mechanismen
Bildverarbeitung: OpenCV, PIL
Datenanalyse: NumPy, scikit-learn, matplotlib, seaborn
Social Media: Reddit API


Installation
Voraussetzungen
Python 3.8+
pip


Setup
# Repository klonen
git clone [repository-url]
cd #github_projektname

# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt


Verwendung
1. Datenvorbereitung
# Datenverarbeitung und Augmentierung
python data_preprocessing.py


2. Modelltraining
# Custom CNN trainieren
python train_model.py

# Alternative Modelle trainieren
python models/vggface_emotion.py


3. Modellvergleich
# Evaluation und Vergleich aller Modelle
python notebooks/Modellvergleich.py


4. Webanwendung starten
# Flask-App starten
python app.py


#Projektstruktur

├── app.py                  → Flask-Backend mit API-Endpunkten für Emotionserkennung
├── data_preprocessing.py   → Laden, Skalieren, Augmentieren und Balancieren des Datensatzes (SMOTE)
├── train_model.py          → Training des CNN inkl. Early Stopping, Checkpointing
├── model_evaluation.py     → Modellbewertung mit Confusion Matrix & Metriken
├── identify_mislabeled.py  → Aufspüren falsch gelabelter Trainingsdaten
├── hyperparameter_optimization.py → GridSearch zur Optimierung der Hyperparameter
├── reddit_api.py           → Abruf von Bilddaten aus Reddit
├── reddit_config.py        → Zugangsdaten & Subreddit-Konfiguration
├── models/
│   ├── vggface_emotion.py        → VGGFace-Modell (Baseline)
│   ├── emotion_model.py          → Definition des Custom CNN (inkl. Residual + Attention)
├── notebooks/
│   └── Modellvergleich.py        → Notebook zur Evaluation & Gegenüberstellung der Modelle
├── templates/
│   ├── index.html                → Upload-Frontend für Einzelbilder
│   └── reddit_feed.html          → Anzeige von Reddit-Ergebnissen
├── tests/
│   ├── test_emotion_model.py     → Unittest für CNN-Architektur
│   └── test_api_endpoints.py     → Test für REST-Endpunkte
├── data/
│   ├── train/                    → Rohdaten (Training)
│   ├── test/                     → Rohdaten (Test)
│   └── processed/                → Vorverarbeitete Daten (.npy-Dateien)



Ergebnisse
Modellvergleich
Custom CNN: 40% Accuracy (beste Leistung)
VGGFace: 24,7% Accuracy


Besonderheiten
Integration von Residual-Blöcken und Attention-Mechanismen
SMOTE für Klassenbalancierung
Umfangreiche Datenaugmentierung
Social-Media-Integration (Reddit statt Instagram aufgrund API-Restriktionen)


Technische Details
Modellarchitekturen
Custom CNN: Residual-Blöcke, Squeeze-and-Excitation, Attention-Mechanismen
VGGFace: VGG-inspirierte Architektur
Input: 48x48x1 Graustufenbilder
Output: 7 Emotionsklassen
Datenverarbeitung
Face Detection mit OpenCV
Bildskalierung und Normalisierung
Datenaugmentierung (Flip, Rotation, Zoom, Translation)
SMOTE für Klassenbalancierung


Lizenz
Dieses Projekt wurde im Rahmen eines Hochschulprojekts entwickelt.


Autor
Jan Serebranski
IU Hochschule
