import cv2
import easyocr
from ultralytics import YOLO
import os
import time
import re
import shutil
import json
from datetime import datetime

# Import your settings
import config

# Ensure output directories exist
os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)

def initialize_model():
    print(f"Loading Models...")
    model = YOLO(config.MODEL_NAME)
    reader = easyocr.Reader(['en'], gpu=True) 
    print(f"Watching: {config.WATCH_FOLDER}")
    return model, reader

def log_to_json(plate_text):
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
    last_size = -1
    for _ in range(15):
        try:
            size = os.path.getsize(filepath)
            if size == last_size and size > 0: return True
            last_size = size
            time.sleep(0.5)
        except OSError: time.sleep(0.5)
    return False

def preprocess_for_ocr(img):
    h, w = img.shape[:2]
    
    # Use settings from config
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
    Intelligent cleaning for 8-char errors on 7-char plates.
    """
    text = re.sub(r'[^A-Z0-9]', '', text)
    danish_pattern = r'^[A-Z]{2}[0-9]{5}$'

    if len(text) == 7: return text

    if len(text) == 8:
        # Phantom '1' on RIGHT
        if text[-1] in ['1', 'I'] and re.match(danish_pattern, text[:-1]):
            return text[:-1]
        # Phantom 'L' on LEFT
        if re.match(danish_pattern, text[1:]):
            return text[1:]

    return text

def process_image(model, reader, full_path, filename):
    img = cv2.imread(full_path)
    if img is None: return

    results = model.predict(source=img, conf=0.25, verbose=False)
    found_plate = False
    detected_text_for_log = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            clean_plate = preprocess_for_ocr(plate_crop)

            ocr_result = reader.readtext(clean_plate, allowlist=config.ALLOWED_CHARS, paragraph=False)
            if not ocr_result: continue

            # Height Filter
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
            final_text = clean_plate_text(detected_text)

            if 5 <= len(final_text) <= 9:
                print(f"✅ PLATE: {final_text}")
                detected_text_for_log = final_text
                found_plate = True
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, final_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow("Debug: Plate View", clean_plate)

    if found_plate:
        log_to_json(detected_text_for_log)
        if img.shape[1] > 1000:
            scale = 1000 / img.shape[1]
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        cv2.imshow("Live Feed", img)
        cv2.waitKey(2000) 
    else:
        print(f"❌ No clear plate found in {filename}")

    try:
        shutil.move(full_path, os.path.join(config.PROCESSED_FOLDER, filename))
        print(f"📂 Moved to processed folder.\n")
    except Exception as e:
        print(f"⚠️ Could not move file: {e}")

# ... (imports remain the same) ...

def main():
    model, reader = initialize_model()
    
    while True:
        try:
            # --- CHANGE IS HERE ---
            # We now use the list from config.py
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
            print(f"Error: {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()