import os
from dotenv import load_dotenv

# Lade Umgebungsvariablen aus .env Datei
load_dotenv()

REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '5TXxbBAjxXUZzFGGmaCsAKA')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '1qeiYh5aoV8tdkhPc-Npvxt-g73bg')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'python:ImageAnalyzer:1.0 (by u/ShotLawfulness9675)')

# Instagram API Konfiguration
INSTAGRAM_APP_ID = os.getenv('INSTAGRAM_APP_ID')
INSTAGRAM_APP_SECRET = os.getenv('INSTAGRAM_APP_SECRET')
INSTAGRAM_REDIRECT_URI = os.getenv('INSTAGRAM_REDIRECT_URI', 'http://localhost:5000/instagram/callback')

# Instagram API Endpunkte
INSTAGRAM_AUTH_URL = 'https://api.instagram.com/oauth/authorize'
INSTAGRAM_TOKEN_URL = 'https://api.instagram.com/oauth/access_token'
INSTAGRAM_GRAPH_URL = 'https://graph.instagram.com'

# Berechtigungen
INSTAGRAM_SCOPES = [
    'user_profile',
    'user_media'
]

# Cache-Konfiguration
CACHE_TYPE = 'simple'
CACHE_DEFAULT_TIMEOUT = 300 