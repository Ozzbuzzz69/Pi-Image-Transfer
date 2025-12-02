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
    6. Logs DETAILED result to Local JSON File.
    7. Sends SIMPLIFIED result to Remote API (plate + timestamp).
    8. Moves file to 'processed' folder.
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

def log_to_json(plate_data):
    """
    Appends the detection data object to the local log file in NDJSON format.
    Args:
        plate_data (dict): A dictionary containing timestamp, filename, plate, etc.
    """
    try:
        with open(config.LOG_FILE, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(plate_data) + "\n")
        print(f"Logged to file: {plate_data['plate_number']}")
    except Exception as e:
        print(f"Error local logging: {e}")

def send_to_api(plate_data):
    """
    Sends the simplified detection data to a remote REST API via HTTP POST.
    
    Args:
        plate_data (dict): Contains only {'plate': '...', 'timestamp': '...'}
    """
    # 1. Check Config
    if not config.ENABLE_API:
        return

    try:
        # 2. Send Request
        response = requests.post(
            config.API_URL, 
            json=plate_data, 
            headers=config.API_HEADERS,
            timeout=5 
        )
        
        # 3. Check Response
        if response.status_code in [200, 201]:
            print(f"Sent to API: {response.status_code}")
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"API Connection Failed: {e}")

def wait_for_write_finish(filepath):
    """
    Prevents reading a file while it is still being copied/written.
    """
    last_size = -1
    for _ in range(15): 
        try:
            size = os.path.getsize(filepath)
            if size == last_size and size > 0: return True
            last_size = size
            time.sleep(0.5)
        except OSError: 
            time.sleep(0.5)
    return False

def preprocess_for_ocr(img):
    """
    Prepares the cropped plate image for the OCR engine.
    """
    h, w = img.shape[:2]
    crop_x = int(w * config.CROP_X_PERCENT) 
    crop_y = int(h * config.CROP_Y_PERCENT)
    img = img[crop_y:h-crop_y, crop_x:w-crop_x]

    img = cv2.resize(img, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return cv2.copyMakeBorder(
        gray, 
        config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, 
        cv2.BORDER_CONSTANT, value=255
    )

def clean_plate_text(text):
    """
    Applies Logic to clean OCR errors, specifically for Danish/EU plates.
    """
    text = re.sub(r'[^A-Z0-9]', '', text)
    danish_pattern = r'^[A-Z]{2}[0-9]{5}$'

    if len(text) == 7: return text

    if len(text) == 8:
        if text[-1] in ['1', 'I'] and re.match(danish_pattern, text[:-1]):
            return text[:-1]
        if re.match(danish_pattern, text[1:]):
            return text[1:]

    return text

def process_image(model, reader, full_path, filename):
    """
    Main Logic Pipeline:
    Load Image -> Detect Object -> Crop & Preprocess -> OCR -> Filter -> Log/API -> Display
    """
    img = cv2.imread(full_path)
    if img is None: return

    # 1. Run YOLO Detection
    results = model.predict(source=img, conf=0.25, verbose=False)
    
    found_plate = False
    
    # We will populate these if a plate is found
    local_log_data = None
    api_payload = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            clean_plate = preprocess_for_ocr(plate_crop)

            # 2. Run OCR
            ocr_result = reader.readtext(clean_plate, allowlist=config.ALLOWED_CHARS, paragraph=False)
            if not ocr_result: continue

            # 3. Height Filtering
            max_height = 0
            valid_parts = []
            for res in ocr_result:
                if res[2] > config.MIN_CONFIDENCE:
                    box = res[0]
                    height = box[2][1] - box[0][1]
                    if height > max_height: max_height = height
                    valid_parts.append((res, height))

            final_parts = []
            for res, height in valid_parts:
                if height > (max_height * config.HEIGHT_RATIO_FILTER):
                    final_parts.append(res)

            final_parts.sort(key=lambda x: x[0][0][0])
            detected_text = "".join([part[1] for part in final_parts])
            
            # 4. Cleaning
            plate_text = clean_plate_text(detected_text)

            # 5. Validation
            if 5 <= len(plate_text) <= 9:
                print(f"✅ PLATE: {plate_text}")
                found_plate = True
                
                # --- PREPARE DATA ---
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # A. Detailed Data (For Local PC Log)
                local_log_data = {
                    "timestamp": current_time,
                    "filename": filename,
                    "plate_number": plate_text,
                    "status": "detected"
                }

                # B. Simplified Data (For REST API)
                api_payload = {
                    "timestamp": current_time,
                    "plate": plate_text
                }
                
                if config.SHOW_GUI:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, plate_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Debug: Plate View", clean_plate)

    # 6. Reporting
    if found_plate and local_log_data:
        # Log full details to disk
        log_to_json(local_log_data)
        
        # Send only timestamp + plate to API
        send_to_api(api_payload)
        
        if config.SHOW_GUI:
            if img.shape[1] > 1000:
                scale = 1000 / img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            cv2.imshow("Live Feed", img)
            cv2.waitKey(2000) 
    else:
        print(f"❌ No clear plate found in {filename}")

    # 7. Cleanup
    try:
        shutil.move(full_path, os.path.join(config.PROCESSED_FOLDER, filename))
        print(f"📂 Moved to processed folder.\n")
    except Exception as e:
        print(f"⚠️ Could not move file: {e}")


def main():
    model, reader = initialize_model()
    
    while True:
        try:
            files = [f for f in os.listdir(config.WATCH_FOLDER) if f.lower().endswith(config.VALID_EXTENSIONS)]
            
            for filename in files:
                full_path = os.path.join(config.WATCH_FOLDER, filename)
                print(f"\n📥 Processing: {filename}")
                
                if wait_for_write_finish(full_path):
                    process_image(model, reader, full_path, filename)
                else:
                    print("Timeout waiting for file.")

            if cv2.waitKey(10) & 0xFF == ord('q'): break
            time.sleep(0.5)

        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()