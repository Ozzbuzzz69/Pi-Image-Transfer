import os

# --- PATHS ---
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
BASE_FOLDER = os.path.join(DESKTOP_PATH, "Pi Image Transfer")

WATCH_FOLDER = os.path.join(BASE_FOLDER, "received_images")
PROCESSED_FOLDER = os.path.join(BASE_FOLDER, "processed_images")
LOG_FILE = os.path.join(BASE_FOLDER, "plate_log.txt")

# --- FILE SETTINGS ---
# Updated to include BMP, TIFF, WEBP
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')

# --- MODEL SETTINGS ---
MODEL_NAME = "license_plate_detector.pt"
ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# --- IMAGE TUNING ---
CROP_X_PERCENT = 0.06  
CROP_Y_PERCENT = 0.02  
RESIZE_SCALE = 2.0     
PADDING_SIZE = 30      

# --- FILTERING ---
MIN_CONFIDENCE = 0.3         
HEIGHT_RATIO_FILTER = 0.6

# --- DISPLAY SETTINGS ---
# Set to True if you want to see the windows (Debugging)
# Set to False if you want it to run silently in the background (Production)
SHOW_GUI = True