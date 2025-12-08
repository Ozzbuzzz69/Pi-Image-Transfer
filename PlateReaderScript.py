"""
===============================================================================
FILE: PlateReaderScript.py
TITLE: Automated License Plate Recognition (ALPR) System
AUTHOR: [Your Name]
DATE: 2025-12-08

DESCRIPTION:
    This script serves as the "Brain" of the vehicle monitoring system.
    It performs the following continuous workflow:
    1.  WATCH: Monitors a specific folder for new image files.
    2.  VERIFY: Waits for file transfer to complete (file write locking).
    3.  DETECT: Uses YOLOv8 AI to locate license plates in the image.
    4.  READ: Uses EasyOCR to extract alphanumeric text from the plate.
    5.  VALIDATE: Checks text against strict Country/Region patterns (from config).
    6.  UPLOAD: Sends valid data to a remote C# REST API.
    7.  LOG: Saves all events to a local text file.
    8.  SORT: Moves images to specific folders (Processed, Ignored, Unidentified).

DEPENDENCIES:
    - opencv-python (cv2): Image manipulation.
    - easyocr: Optical Character Recognition.
    - ultralytics: YOLOv8 Object Detection.
    - requests: HTTP API communication.
    - config.py: External configuration file.

USAGE:
    Run this script in a terminal. It will loop indefinitely until interrupted
    with CTRL+C.
===============================================================================
"""

import cv2
import easyocr
from ultralytics import YOLO
import os
import time
import re
import shutil
import json
import requests
from datetime import datetime

# Import configuration from config.py
# This decouples logic from settings, allowing easy tuning without code changes.
import config 

# =============================================================================
# INITIAL SETUP
# =============================================================================
# Create necessary directories immediately on startup.
# This prevents the script from crashing later if a folder is missing.
os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)
os.makedirs(config.UNIDENTIFIED_FOLDER, exist_ok=True)
os.makedirs(config.IGNORED_FOLDER, exist_ok=True) 

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def initialize_model():
    """
    PURPOSE:
        Loads the heavy AI models into RAM at system startup.
    
    WHY:
        Loading YOLO and EasyOCR takes 5-10 seconds. We do this ONCE 
        so we can process subsequent images in milliseconds.
    
    RETURNS:
        model (YOLO): The object detection model.
        reader (easyocr.Reader): The text recognition engine.
    """
    print(f"--- SYSTEM STARTUP ---")
    print(f"1. Loading YOLOv8 Model ({config.MODEL_NAME})...")
    model = YOLO(config.MODEL_NAME)
    
    print(f"2. Loading EasyOCR Engine...")
    # 'gpu=True' significantly speeds up processing if NVIDIA CUDA is available.
    reader = easyocr.Reader(config.OCR_LANGUAGES, gpu=config.USE_GPU) 
    
    print(f"3. System Ready. Watching: {config.WATCH_FOLDER}")
    return model, reader

def log_to_json(plate_data):
    """
    PURPOSE:
        Appends a detection record to a local log file.
    
    ARGS:
        plate_data (dict): Dictionary containing timestamp, plate, status, etc.
        
    NOTE:
        Uses NDJSON format (Newline Delimited JSON). This is safer than a 
        standard JSON array because if the script crashes, the file remains valid.
    """
    try:
        with open(config.LOG_FILE, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(plate_data) + "\n\n")
        print(f"✅ Logged to local file: {plate_data['plate_number']}")
    except Exception as e:
        print(f"⚠️ Error writing to local log: {e}")

def send_to_api(plate_data):
    """
    PURPOSE:
        Transmits the plate data to the remote REST API (C# Backend).
    
    ARGS:
        plate_data (dict): Payload matching the API's Data Transfer Object (DTO).
                           Expected keys: 'time', 'Plate'.
    
    BEHAVIOR:
        - Skips execution if config.ENABLE_API is False.
        - Uses a 5-second timeout to prevent freezing on network loss.
    """
    if not config.ENABLE_API: return

    try:
        response = requests.post(
            config.API_URL, 
            json=plate_data, 
            headers=config.API_HEADERS,
            timeout=5 
        )
        if response.status_code in [200, 201]:
            print(f"🚀 Sent to API successfully: {response.status_code}")
        else:
            print(f"⚠️ API Rejected Data: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API Connection Failed: {e}")

def wait_for_write_finish(filepath):
    """
    PURPOSE:
        Ensures a file is fully copied before attempting to read it.
    
    THE PROBLEM:
        When a camera sends an image, the OS creates the filename immediately (0 bytes),
        then slowly fills it with data. Reading it too early causes a crash.
    
    LOGIC:
        Checks file size every X seconds. If size > 0 and hasn't changed 
        between checks, the write is complete.
    """
    last_size = -1
    for _ in range(config.FILE_CHECK_RETRIES): 
        try:
            size = os.path.getsize(filepath)
            # Stability Check: Size is non-zero and hasn't changed since last check
            if size == last_size and size > 0: return True
            
            last_size = size
            time.sleep(config.FILE_CHECK_DELAY)
        except OSError: 
            # File might be locked by OS, wait and retry
            time.sleep(config.FILE_CHECK_DELAY)
    return False

def preprocess_for_ocr(img):
    """
    PURPOSE:
        Optimizes a raw image crop for the OCR engine.
        Includes CRITICAL SAFETY CHECKS to prevent crashing on noise.
    
    PIPELINE:
        1. Size Check: Reject images smaller than 15x15px (prevents resize crash).
        2. Crop: Remove percentage of edges (removes license plate frames).
        3. Resize: Zoom in (cubic interpolation) to make text clear.
        4. Grayscale: Remove color noise.
        5. Padding: Add white border so text doesn't touch the edge.
        
    RETURNS:
        Mat (Image) or None (if image was invalid/too small).
    """
    # 1. Basic Validity Check
    if img is None or img.size == 0: return None

    h, w = img.shape[:2]
    
    # 2. Skip Tiny Objects (Noise/Background cars)
    # If YOLO detects a car 100 meters away, it's too small to read. Skip it.
    if h < 15 or w < 15: 
        return None

    crop_x = int(w * config.CROP_X_PERCENT) 
    crop_y = int(h * config.CROP_Y_PERCENT)
    
    # 3. Prevent Over-Cropping
    # If cropping 5% removes the whole image (on small boxes), skip the crop.
    if (2 * crop_x >= w) or (2 * crop_y >= h):
        pass # Keep original
    else:
        img = img[crop_y:h-crop_y, crop_x:w-crop_x]

    # 4. Final Empty Check
    if img.size == 0: return None

    # 5. Resize (Zoom)
    try:
        img = cv2.resize(img, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE, interpolation=cv2.INTER_CUBIC)
    except cv2.error:
        # Failsafe for rare OpenCV errors
        return None
    
    # 6. Color Processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 7. Add Border
    return cv2.copyMakeBorder(
        gray, 
        config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, 
        cv2.BORDER_CONSTANT, value=255
    )

def clean_plate_text(text):
    """
    PURPOSE:
        Corrects OCR errors by comparing text against valid Config Patterns.
        
    ALGORITHM:
        1. Strip non-alphanumeric chars.
        2. Check if it matches a pattern (e.g., Danish). If yes, return.
        3. If no, try removing the last char (fixes edge noise ']').
        4. If no, try removing the first char (fixes edge noise '|').
    """
    # Step 1: Remove special chars (keep only A-Z and 0-9)
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Bypass if pattern enforcement is disabled
    if not config.ENFORCE_REGION_PATTERNS: return text

    # Helper function to check config dictionary
    def is_valid(candidate):
        for pattern in config.ALLOWED_PATTERNS.values():
            if re.match(pattern, candidate): return True
        return False

    # Step 2: Perfect match check
    if is_valid(text): return text

    # Step 3: Heuristic Fixing (Trimming noise)
    if len(text) > 4:
        if is_valid(text[:-1]): return text[:-1] # Trim Suffix
        if is_valid(text[1:]): return text[1:]   # Trim Prefix

    return text

def process_image(model, reader, full_path, filename):
    """
    PURPOSE:
        The main logic pipeline for a SINGLE image file.
    
    STEPS:
        1. Detect Objects (YOLO).
        2. Loop through every detected box (Multiple plates supported).
        3. Preprocess & Safety Check (Skip tiny noise).
        4. Read Text (OCR).
        5. Validate Text (Regex & Length).
        6. Queue results (Valid -> API, Invalid -> Log).
        7. Sort File (Move to appropriate folder).
    """
    img = cv2.imread(full_path)
    if img is None: return

    # YOLO Inference
    results = model.predict(source=img, conf=config.YOLO_CONFIDENCE, verbose=False)
    
    # Storage for this specific image
    all_logs = []       
    valid_payloads = [] 
    found_any_plate = False

    # Iterate over detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            
            # --- CRITICAL SAFETY CHECK ---
            clean_plate = preprocess_for_ocr(plate_crop)
            if clean_plate is None:
                # If preprocessing returned None, the box was noise. Skip it.
                continue 
            # -----------------------------

            # OCR Reading
            ocr_result = reader.readtext(clean_plate, allowlist=config.ALLOWED_CHARS, paragraph=False)
            if not ocr_result: continue

            # Geometry Filter: Remove small text fragments compared to main text
            max_height = 0
            valid_parts = []
            for res in ocr_result:
                if res[2] > config.MIN_CONFIDENCE:
                    box_coords = res[0]
                    height = box_coords[2][1] - box_coords[0][1]
                    if height > max_height: max_height = height
                    valid_parts.append((res, height))

            final_parts = []
            for res, height in valid_parts:
                if height > (max_height * config.HEIGHT_RATIO_FILTER):
                    final_parts.append(res)

            # Reconstruct string
            final_parts.sort(key=lambda x: x[0][0][0])
            detected_text = "".join([part[1] for part in final_parts])
            
            # Text Cleaning
            plate_text = clean_plate_text(detected_text)

            # Length Validation
            if config.MIN_PLATE_LENGTH <= len(plate_text) <= config.MAX_PLATE_LENGTH:
                print(f"✨ DETECTED: {plate_text}")
                found_any_plate = True
                
                current_time = datetime.now().isoformat(timespec='seconds')
                log_entry = { "timestamp": current_time, "filename": filename, "plate_number": plate_text, "status": "detected" }
                all_logs.append(log_entry)

                # Pattern Validation (Region Check)
                is_valid_pattern = False
                def check_pattern(txt):
                    if not config.ENFORCE_REGION_PATTERNS: return True
                    for pattern in config.ALLOWED_PATTERNS.values():
                        if re.match(pattern, txt): return True
                    return False

                if check_pattern(plate_text):
                    print(f"✅ Config Match: '{plate_text}' -> API Queue.")
                    valid_payloads.append({ "time": current_time, "Plate": plate_text })
                    is_valid_pattern = True
                else:
                    print(f"🚫 Config Reject: '{plate_text}' -> Local Log only.")

                # Draw Visuals (GUI)
                if config.SHOW_GUI:
                    color = (0, 255, 0) if is_valid_pattern else (0, 0, 255) # Green=API, Red=LogOnly
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    # Output: Write Logs & Send API Requests
    for entry in all_logs: log_to_json(entry)
    for payload in valid_payloads: send_to_api(payload)

    # Output: Move File
    try:
        if len(valid_payloads) > 0:
            dest_folder = config.PROCESSED_FOLDER
            print(f"📂 Success. Moving to PROCESSED.\n")
        elif found_any_plate:
            dest_folder = config.IGNORED_FOLDER
            print(f"📂 Rejected. Moving to IGNORED.\n")
        else:
            dest_folder = config.UNIDENTIFIED_FOLDER
            print(f"❌ Empty. Moving to UNIDENTIFIED.\n")

        shutil.move(full_path, os.path.join(dest_folder, filename))
    except Exception as e:
        print(f"🔥 FILE MOVE ERROR: {e}")

    # Display Result
    if config.SHOW_GUI and found_any_plate:
        # Resize huge images for screen
        if img.shape[1] > 1000:
            scale = 1000 / img.shape[1]
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        cv2.imshow("Live Feed", img)
        cv2.waitKey(config.DISPLAY_DURATION_MS)

def main():
    """
    Entry Point. Initializes resources and runs the infinite watch loop.
    """
    model, reader = initialize_model()
    print("--- MAIN LOOP STARTED ---")
    
    while True:
        try:
            # Check folder for files matching allowed extensions
            files = [f for f in os.listdir(config.WATCH_FOLDER) if f.lower().endswith(config.VALID_EXTENSIONS)]
            
            for filename in files:
                full_path = os.path.join(config.WATCH_FOLDER, filename)
                print(f"\n📥 Processing: {filename}")
                
                # Ensure file is ready
                if wait_for_write_finish(full_path):
                    process_image(model, reader, full_path, filename)
                else:
                    print("⛔ Skipped (Write Timeout)")
            
            # Pause to save CPU cycles
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by User.")
            break
        except Exception as e:
            print(f"Main Loop Error: {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()