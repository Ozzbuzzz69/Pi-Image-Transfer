"""
===============================================================================
FILE: PlateReaderScript.py
DESCRIPTION: 
    The core License Plate Recognition (LPR) engine.
    
WORKFLOW:
    1. Watches a specific folder for new images.
    2. Detects license plates using YOLOv8.
    3. Preprocesses the plate (Crop -> Zoom -> Grayscale -> Pad).
    4. Reads text using EasyOCR.
    5. Applies Cleaning Logic (Danish/EU standards, Height Filtering).
    6. Logs result to JSON and moves file to 'processed' folder.
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
from datetime import datetime

# Import configuration from config.py
import config

# Ensure output directories exist immediately
os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)

def initialize_model():
    """
    Loads the AI models into memory.
    Returns:
        model (YOLO): The object detector for finding plates.
        reader (EasyOCR): The text reader.
    """
    print(f"Loading Models...")
    model = YOLO(config.MODEL_NAME)
    # gpu=True enables CUDA if available, otherwise falls back to CPU
    reader = easyocr.Reader(['en'], gpu=True) 
    print(f"Watching: {config.WATCH_FOLDER}")
    return model, reader

def log_to_json(plate_text):
    """
    Appends the detected plate to the log file in NDJSON format.
    Format: {"timestamp": "...", "plate_number": "...", "status": "detected"}
    """
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plate_number": plate_text,
        "status": "detected"
    }
    try:
        with open(config.LOG_FILE, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(data) + "\n")
        print(f"📝 Logged: {plate_text}")
    except Exception as e:
        print(f"❌ Error logging: {e}")

def wait_for_write_finish(filepath):
    """
    Prevents reading a file while it is still being copied/written.
    Waits until file size stops changing for 0.5 seconds.
    """
    last_size = -1
    for _ in range(15): # Max wait approx 7.5 seconds
        try:
            size = os.path.getsize(filepath)
            # If size > 0 and hasn't changed since last check, it's safe.
            if size == last_size and size > 0: return True
            last_size = size
            time.sleep(0.5)
        except OSError: 
            time.sleep(0.5)
    return False

def preprocess_for_ocr(img):
    """
    Prepares the cropped plate image for the OCR engine.
    
    Steps:
    1. Crop Edges: Removes plastic frames/borders.
    2. Resize: Zooms in (2x) using Cubic Interpolation.
    3. Grayscale: Removes color noise.
    4. Padding: Adds a white border so characters aren't on the edge.
    """
    h, w = img.shape[:2]
    
    # Calculate crop amount based on config percentages
    crop_x = int(w * config.CROP_X_PERCENT) 
    crop_y = int(h * config.CROP_Y_PERCENT)
    
    # Apply Crop
    img = img[crop_y:h-crop_y, crop_x:w-crop_x]

    # Resize/Zoom
    img = cv2.resize(img, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE, interpolation=cv2.INTER_CUBIC)
    
    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Add Padding (White Border)
    return cv2.copyMakeBorder(
        gray, 
        config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, 
        cv2.BORDER_CONSTANT, value=255
    )

def clean_plate_text(text):
    """
    Applies Logic to clean OCR errors, specifically for Danish/EU plates.
    
    The Logic:
    1. Removes all non-alphanumeric characters.
    2. Handles '8-Character' errors where a 7-char plate gets an extra '1' or 'L'.
       - Checks Left side (Phantom 'L')
       - Checks Right side (Phantom '1')
       - Verifies against Danish pattern (2 Letters + 5 Numbers).
    """
    # 1. Basic Cleaning (Allow A-Z, 0-9)
    text = re.sub(r'[^A-Z0-9]', '', text)
    
    # Regex for standard Danish plate: 2 Letters followed by 5 Numbers
    danish_pattern = r'^[A-Z]{2}[0-9]{5}$'

    # If it's already perfect, return it
    if len(text) == 7: return text

    # Handle common error where borders are read as characters (making length 8)
    if len(text) == 8:
        # Case A: Phantom '1' or 'I' on the RIGHT side (e.g. AB123451)
        if text[-1] in ['1', 'I'] and re.match(danish_pattern, text[:-1]):
            return text[:-1]
        
        # Case B: Phantom 'L' on the LEFT side (e.g. LAB12345)
        # This often happens with the blue EU strip
        if re.match(danish_pattern, text[1:]):
            return text[1:]

    return text

def process_image(model, reader, full_path, filename):
    """
    Main Logic Pipeline:
    Load Image -> Detect Object -> Crop & Preprocess -> OCR -> Filter -> Display
    """
    img = cv2.imread(full_path)
    if img is None: return

    # 1. Run YOLO Detection
    results = model.predict(source=img, conf=0.25, verbose=False)
    
    found_plate = False
    detected_text_for_log = None

    for result in results:
        for box in result.boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the detected plate
            plate_crop = img[y1:y2, x1:x2]
            
            # Preprocess the crop
            clean_plate = preprocess_for_ocr(plate_crop)

            # 2. Run OCR
            ocr_result = reader.readtext(clean_plate, allowlist=config.ALLOWED_CHARS, paragraph=False)
            if not ocr_result: continue

            # 3. Height Filtering (Geometry Check)
            # This rejects text that is significantly smaller than the main plate number (e.g., "DK")
            max_height = 0
            valid_parts = []
            
            # Pass 1: Find Max Height
            for res in ocr_result:
                if res[2] > config.MIN_CONFIDENCE:
                    box = res[0]
                    height = box[2][1] - box[0][1]
                    if height > max_height: max_height = height
                    valid_parts.append((res, height))

            # Pass 2: Keep only text close to max height
            final_parts = []
            for res, height in valid_parts:
                if height > (max_height * config.HEIGHT_RATIO_FILTER):
                    final_parts.append(res)

            # 4. Stitching
            # Sort parts left-to-right based on X coordinate
            final_parts.sort(key=lambda x: x[0][0][0])
            detected_text = "".join([part[1] for part in final_parts])
            
            # 5. Cleaning
            final_text = clean_plate_text(detected_text)

            # 6. Validation (Length Check)
            if 5 <= len(final_text) <= 9:
                print(f"PLATE: {final_text}")
                detected_text_for_log = final_text
                found_plate = True
                
                # Draw Graphics (Only if GUI is enabled)
                if config.SHOW_GUI:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, final_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Debug: Plate View", clean_plate)

    # 7. Post-Processing Actions
    if found_plate:
        log_to_json(detected_text_for_log)
        
        if config.SHOW_GUI:
            # Resize image for display if it's too huge
            if img.shape[1] > 1000:
                scale = 1000 / img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            cv2.imshow("Live Feed", img)
            cv2.waitKey(2000) 
    else:
        print(f"No clear plate found in {filename}")

    # 8. Cleanup: Move file to 'processed' folder
    try:
        shutil.move(full_path, os.path.join(config.PROCESSED_FOLDER, filename))
        print(f"Moved to processed folder.\n")
    except Exception as e:
        print(f"Could not move file: {e}")


def main():
    """
    Continuous loop that watches the folder for new files.
    """
    model, reader = initialize_model()
    
    while True:
        try:
            # Scan folder for files ending with valid extensions
            files = [f for f in os.listdir(config.WATCH_FOLDER) if f.lower().endswith(config.VALID_EXTENSIONS)]
            
            for filename in files:
                full_path = os.path.join(config.WATCH_FOLDER, filename)
                
                print(f"\n📥 Processing: {filename}")
                
                # Wait for file to finish downloading/writing before opening
                if wait_for_write_finish(full_path):
                    process_image(model, reader, full_path, filename)
                else:
                    print("Timeout waiting for file.")

            # Press 'q' to quit manually
            if cv2.waitKey(10) & 0xFF == ord('q'): break
            time.sleep(0.5)

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()