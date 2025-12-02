"""
===============================================================================
FILE: config.py
DESCRIPTION: 
    Central configuration file for the License Plate Recognition System.
    This file holds all tunable parameters, file paths, and model settings.
    Adjusting values here changes the behavior of the main script without
    needing to touch the logic code.
===============================================================================
"""

import os

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================
# Define the base directory on the User's Desktop
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
BASE_FOLDER = os.path.join(DESKTOP_PATH, "Pi Image Transfer")

# Input folder: Where ImgReciever saves images
WATCH_FOLDER = os.path.join(BASE_FOLDER, "received_images")

# Archive folder: Where images are moved after successful processing
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "processed_images")

# Log file: The NDJSON text file where detection results are appended
LOG_FILE = os.path.join(BASE_FOLDER, "plate_log.txt")


# =============================================================================
# 2. FILE & MODEL SETTINGS
# =============================================================================
# Tuple of allowed image formats. Files not matching these are ignored.
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

# The filename of the YOLOv8 model (must be in the same directory as the script)
MODEL_NAME = "license_plate_detector.pt"

# The Allowlist for OCR. We only want Uppercase Letters and Numbers.
ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


# =============================================================================
# 3. IMAGE PREPROCESSING TUNING ("THE SWEET SPOTS")
# =============================================================================
# These values are critical for accuracy. They are tuned for standard
# camera angles to remove license plate frames that confuse OCR.

# Percentage to crop from Left/Right to remove plate holders/frames
# 0.06 = 6%. Removes "Phantom 1" errors caused by vertical frame edges.
CROP_X_PERCENT = 0.06  

# Percentage to crop from Top/Bottom
CROP_Y_PERCENT = 0.02  

# Scale factor to zoom image. 2.0 = 200%.
# Larger text is easier for OCR to read.
RESIZE_SCALE = 2.0     

# Pixels of white border to add around the plate.
# Prevents text touching the edge from being cut off.
PADDING_SIZE = 30      


# =============================================================================
# 4. LOGIC FILTERS
# =============================================================================
# Minimum confidence (0.0 to 1.0) for EasyOCR to accept a text block.
MIN_CONFIDENCE = 0.3         

# Geometry Filter: Rejects text that is too small compared to the main plate.
# 0.6 means: "If text height is less than 60% of the tallest text, delete it."
# This effectively removes Country Codes (e.g., small "DK") and dealer text.
HEIGHT_RATIO_FILTER = 0.6


# =============================================================================
# 5. DISPLAY SETTINGS
# =============================================================================
# Set to True: Shows pop-up windows with the detected plate (Good for Debugging).
# Set to False: Runs silently in the background (Good for 24/7 Production).
SHOW_GUI = True


# =============================================================================
# 6. REST API CONFIGURATION
# =============================================================================
# Set to True to send detection results to a remote server/API.
ENABLE_API = False  # <--- Set to True when you have a server ready

# The URL where the POST request will be sent
API_URL = "http://YOUR_SERVER_IP:5000/api/plate"

# Optional Headers (Useful for Authorization tokens or Content-Type)
API_HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": "Bearer YOUR_TOKEN_HERE" 
}