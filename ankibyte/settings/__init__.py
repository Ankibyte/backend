# Add these settings to your existing settings file
import os
from pathlib import Path

# Get the base directory (assuming settings is in ankibyte/settings/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Media files configuration
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# File upload settings
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10MB

# Make sure 'rest_framework' and 'corsheaders' are in INSTALLED_APPS
INSTALLED_APPS = [
    # ... your existing apps ...
    'rest_framework',
    'corsheaders',
    'api',
]

# Add CORS middleware
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    # ... your existing middleware ...
]

# CORS settings
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Your React frontend
]