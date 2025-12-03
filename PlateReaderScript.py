"""
===============================================================================
FILE: PlateReaderScript.py
DESCRIPTION: 
    The core License Plate Recognition (LPR) engine.
    
    This script acts as the "Brain" of the operation. It sits in an infinite
    loop watching a folder. When an image arrives, it ensures the file is 
    fully written, then runs AI detection, OCR reading, and finally sends
    the data to your C# API and local logs.

AUTHOR: [Your Name]
DATE: 2025-12-03
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
# This file holds all our tunable settings (paths, thresholds, URLs)
import config

# Create the processed folder immediately so we don't crash later
os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)
os.makedirs(config.UNIDENTIFIED_FOLDER, exist_ok=True)

def initialize_model():
    """
    PURPOSE:
        Loads the AI models into RAM at startup.
    
    WHY THIS MATTERS:
        Loading models takes 5-10 seconds. We do this ONCE at the start
        so we don't waste time reloading them for every single car.
    
    RETURNS:
        model (YOLO): The object detector that finds WHERE the plate is.
        reader (EasyOCR): The engine that reads WHAT is written on it.
    """
    print(f"--- SYSTEM STARTUP ---")
    print(f"1. Loading YOLOv8 Model ({config.MODEL_NAME})...")
    model = YOLO(config.MODEL_NAME)
    
    print(f"2. Loading EasyOCR Engine...")
    # 'gpu=True' attempts to use NVIDIA CUDA. If no GPU is found, it silently falls back to CPU.
    reader = easyocr.Reader(['en'], gpu=True) 
    
    print(f"3. System Ready. Watching: {config.WATCH_FOLDER}")
    return model, reader

def log_to_json(plate_data):
    """
    PURPOSE:
        Saves a detailed record of the detection to a local text file.
    
    FORMAT:
        NDJSON (Newline Delimited JSON). Each line is a valid JSON object.
        We add an extra newline ('\\n\\n') to create visual spacing between entries.
    """
    try:
        with open(config.LOG_FILE, mode='a', encoding='utf-8') as f:
            # Dump the dictionary to string, add two newlines for spacing
            f.write(json.dumps(plate_data) + "\n\n")
        print(f"✅ Logged to local file: {plate_data['plate_number']}")
    except Exception as e:
        print(f"⚠️ Error writing to local log: {e}")

def send_to_api(plate_data):
    """
    PURPOSE:
        Sends the simplified data to the C# REST API.
    
    ARGUMENTS:
        plate_data (dict): Must contain keys matching the C# DTO:
                           { "Plate": "string", "time": "ISO-Date" }
    """
    # 1. Safety Check: Is the API feature turned on in config.py?
    if not config.ENABLE_API:
        return

    try:
        # 2. Attempt the HTTP POST request
        # We use a timeout=5 so the script doesn't freeze if the server is down.
        response = requests.post(
            config.API_URL, 
            json=plate_data, 
            headers=config.API_HEADERS,
            timeout=5 
        )
        
        # 3. Check if the server said "OK" (Status 200 or 201)
        if response.status_code in [200, 201]:
            print(f"🚀 Sent to API successfully: {response.status_code}")
        else:
            print(f"⚠️ API Rejected Data: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API Connection Failed (Is the C# server running?): {e}")

def wait_for_write_finish(filepath):
    """
    PURPOSE:
        Prevents the script from crashing by reading a file that is still copying.
        
    HOW IT WORKS (POLLING):
        1. It looks at the file size.
        2. It waits 0.5 seconds.
        3. It looks at the file size again.
        4. IF size is unchanged AND size > 0: The file is ready.
        5. IF size changed: The file is still copying, wait again.
        
    RETURNS:
        True: File is safe to open.
        False: File timed out (took too long or stuck).
    """
    last_size = -1
    # Try 15 times x 0.5s = 7.5 seconds max wait time
    for _ in range(15): 
        try:
            size = os.path.getsize(filepath)
            # The "Golden Condition": Size is stable and not empty
            if size == last_size and size > 0: return True
            
            last_size = size
            time.sleep(0.5)
        except OSError: 
            # If OS locks the file completely, wait and retry
            time.sleep(0.5)
    return False

def preprocess_for_ocr(img):
    """
    PURPOSE:
        Transforms the raw crop of the plate into an image optimized for EasyOCR.
        
    STEPS:
        1. Crop Borders: Removes plastic license plate holders/frames.
        2. Zoom: Enlarges text (2x) so OCR sees it clearly.
        3. Grayscale: Removes color noise (shadows/mud).
        4. Padding: Adds white border so text doesn't touch the edge.
    """
    h, w = img.shape[:2]
    
    # Calculate how many pixels to slice off the edges
    crop_x = int(w * config.CROP_X_PERCENT) 
    crop_y = int(h * config.CROP_Y_PERCENT)
    
    # Perform the crop
    img = img[crop_y:h-crop_y, crop_x:w-crop_x]

    # Resize (Zoom in) using Cubic Interpolation (smoother than linear)
    img = cv2.resize(img, None, fx=config.RESIZE_SCALE, fy=config.RESIZE_SCALE, interpolation=cv2.INTER_CUBIC)
    
    # Convert to black and white
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Add the white border
    return cv2.copyMakeBorder(
        gray, 
        config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, config.PADDING_SIZE, 
        cv2.BORDER_CONSTANT, value=255
    )

def clean_plate_text(text):
    """
    PURPOSE:
        Cleans up OCR mistakes using regex logic.
        Specifically tuned for standard plates (e.g., Danish 2 letters + 5 numbers).
    """
    # Step 1: Remove anything that isn't a Capital Letter or a Number
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Step 2: Define the Danish pattern (AA12345)
    danish_pattern = r'^[A-Z]{2}[0-9]{5}$'

    if len(text) == 7: return text

    # Step 3: Handle "Phantom Character" errors (e.g., edge of frame read as '1' or 'I')
    if len(text) == 8:
        # Check if the last char is noise (e.g., "AB12345I")
        if text[-1] in ['1', 'I'] and re.match(danish_pattern, text[:-1]):
            return text[:-1]
        # Check if the first char is noise (e.g., "1AB12345")
        if re.match(danish_pattern, text[1:]):
            return text[1:]

    return text

def process_image(model, reader, full_path, filename):
    """
    THE MAIN PIPELINE.
    Everything happens here for a single image file.
    """
    # 1. Load the image
    img = cv2.imread(full_path)
    if img is None: return

    # 2. YOLO Detection (Find the plate)
    results = model.predict(source=img, conf=0.12, verbose=False)
    
    found_plate = False
    local_log_data = None
    api_payload = None

    for result in results:
        for box in result.boxes:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop just the license plate area
            plate_crop = img[y1:y2, x1:x2]
            
            # 3. Image Processing (Zoom, Gray, Pad)
            clean_plate = preprocess_for_ocr(plate_crop)

            # 4. Optical Character Recognition (Read Text)
            ocr_result = reader.readtext(clean_plate, allowlist=config.ALLOWED_CHARS, paragraph=False)
            if not ocr_result: continue

            # 5. Logic Filter: Height Ratio
            # Removes small text (like "DK" or dealer names) by comparing text height 
            # to the largest text found.
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
                # If text is less than 60% of the main plate height, discard it.
                if height > (max_height * config.HEIGHT_RATIO_FILTER):
                    final_parts.append(res)

            # Re-assemble the text parts left to right
            final_parts.sort(key=lambda x: x[0][0][0])
            detected_text = "".join([part[1] for part in final_parts])
            
            # 6. Final Text Cleaning
            plate_text = clean_plate_text(detected_text)

            # 7. Validation: Is the result reasonable?
            if 5 <= len(plate_text) <= 9:
                print(f"✨ DETECTED: {plate_text}")
                found_plate = True
                
                # --- PREPARE DATA ---
                # Use ISO Format with seconds (No microseconds) for compatibility
                current_time = datetime.now().isoformat(timespec='seconds')

                # A. Data for Local Log (Detailed)
                local_log_data = {
                    "timestamp": current_time,
                    "filename": filename,
                    "plate_number": plate_text,
                    "status": "detected"
                }

                # B. Data for API (Matches C# PlateDto class)
                # Note: Keys are Case-Sensitive! "Plate" and "time".
                api_payload = {
                    "time": current_time, 
                    "Plate": plate_text
                }
                
                # Visual Debugging (Draw boxes on image)
                if config.SHOW_GUI:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(img, plate_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Debug: Plate View", clean_plate)

    # 8. Output Handling
    if found_plate and local_log_data:
        # Write to text file
        log_to_json(local_log_data)
        
        # Send to server
        send_to_api(api_payload)
        
        # Show image on screen
        if config.SHOW_GUI:
            # If image is 4K/Huge, shrink it so it fits on screen
            if img.shape[1] > 1000:
                scale = 1000 / img.shape[1]
                img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            cv2.imshow("Live Feed", img)
            cv2.waitKey(2000) # Pause for 2 seconds to see result
    else:
        print(f"❌ No clear plate found in {filename}")

    # Intelligent File Sorting Logic
    try:
        if found_plate:
            # If success, go to Processed folder
            dest_folder = config.PROCESSED_FOLDER
            print(f"📂 Moving to PROCESSED folder.\n")
        else:
            # If fail, go to Unidentified folder for manual review
            dest_folder = config.UNIDENTIFIED_FOLDER
            print(f"📂 Moving to UNIDENTIFIED folder.\n")

        shutil.move(full_path, os.path.join(dest_folder, filename))

    except Exception as e:
        print(f"🔥 CRITICAL: Could not move file! {e}")

def main():
    """
    Entry point of the script.
    To stop the script, press CTRL+C in the terminal.
    """
    model, reader = initialize_model()
    
    print("--- STARTING MAIN LOOP (Press CTRL+C to quit) ---")
    
    while True:
        try:
            # 1. Scan folder for valid image files
            files = [f for f in os.listdir(config.WATCH_FOLDER) if f.lower().endswith(config.VALID_EXTENSIONS)]
            
            for filename in files:
                full_path = os.path.join(config.WATCH_FOLDER, filename)
                print(f"\n📥 Incoming File: {filename}")
                
                # 2. Check if file is fully written before processing
                if wait_for_write_finish(full_path):
                    process_image(model, reader, full_path, filename)
                else:
                    print("⛔ Timeout: File took too long to write. Skipping.")

            # 3. Sleep to save CPU
            time.sleep(1)

        except KeyboardInterrupt:
            # This runs when you press CTRL+C
            print("\n🛑 User stopped script via Keyboard. Exiting...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()