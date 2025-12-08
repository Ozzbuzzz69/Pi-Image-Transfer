"""
===============================================================================
FILE: config.py
DESCRIPTION: 
    Central configuration file for the License Plate Recognition System.
===============================================================================
"""

import os

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
BASE_FOLDER = os.path.join(DESKTOP_PATH, "Pi Image Transfer")

WATCH_FOLDER = os.path.join(BASE_FOLDER, "received_images")
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "processed_images")
UNIDENTIFIED_FOLDER = os.path.join(BASE_FOLDER, "unidentified_images")
IGNORED_FOLDER = os.path.join(BASE_FOLDER, "ignored_images")
LOG_FILE = os.path.join(BASE_FOLDER, "plate_log.txt")

# =============================================================================
# 2. FILE & MODEL SETTINGS
# =============================================================================
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
MODEL_NAME = "license_plate_detector.pt"
ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# =============================================================================
# 3. IMAGE PREPROCESSING TUNING
# =============================================================================
CROP_X_PERCENT = 0.045
CROP_Y_PERCENT = 0.011 
RESIZE_SCALE = 4.0     
PADDING_SIZE = 25      

# =============================================================================
# 4. LOGIC FILTERS
# =============================================================================
MIN_CONFIDENCE = 0.20       
HEIGHT_RATIO_FILTER = 0.4

# =============================================================================
# 5. DISPLAY SETTINGS
# =============================================================================
SHOW_GUI = True

# =============================================================================
# 6. REST API CONFIGURATION
# =============================================================================
ENABLE_API = True
API_URL = "https://parkwhererest20251203132035-gdh2hyd0c9ded8ah.germanywestcentral-01.azurewebsites.net/api/parkwhere"
API_HEADERS = { "Content-Type": "application/json" }

# =============================================================================
# 7. REGION & PATTERN CONFIGURATION
# =============================================================================
ENFORCE_REGION_PATTERNS = True
ALLOWED_PATTERNS = {
    "Denmark_Standard": r'^[A-Z]{2}[0-9]{5}$',
    #"Sweden_Standard": r'^[A-Z]{3}[0-9]{2}[A-Z0-9]{1}$',
    #"Norway_Standard": r'^[A-Z]{2}[0-9]{5}$',
    #"Germany_Standard": r'^[A-Z]{1,3}[A-Z]{1,2}[0-9]{1,4}$',
}

# =============================================================================
# 8. ADVANCED AI TUNING (NEW)
# =============================================================================
# Threshold to accept a plate detection. 
# 0.15 = Sensitive (Finds small plates). 0.50 = Strict (Only clear plates).
YOLO_CONFIDENCE = 0.12 
OCR_LANGUAGES = ['en']
USE_GPU = True

# =============================================================================
# 9. VALIDATION LIMITS (NEW)
# =============================================================================
MIN_PLATE_LENGTH = 2
MAX_PLATE_LENGTH = 10

# =============================================================================
# 10. SYSTEM TIMING & SAFETY (NEW)
# =============================================================================
FILE_CHECK_RETRIES = 20
FILE_CHECK_DELAY = 0.5
DISPLAY_DURATION_MS = 2000