import cv2
import easyocr
from ultralytics import YOLO
import os
import time
import re

# --- CONFIGURATION ---
WATCH_FOLDER = r"C:\Users\ijhuigiytr\Desktop\Pi Image Transfer\received_images"
MODEL_NAME = "license_plate_detector.pt"

# Standard alphanumerics are enough for DK, PL, and DE plates.
# (Even if a German plate has an 'Ö', it is usually read as 'O' or requires complex training)
ALLOWED_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ0123456789'

processed_files = {}

def initialize_model():
    print(f"Loading Models...")
    model = YOLO(MODEL_NAME)
    # Using 'en' is best for plates. We don't need 'pl' or 'de' language dictionaries 
    # because license plates are random codes, not dictionary words.
    reader = easyocr.Reader(['en'], gpu=True) 
    print("Waiting for images...")
    return model, reader

def get_file_timestamp(filepath):
    try: return os.path.getmtime(filepath)
    except OSError: return 0

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
    """
    Simple cleaning: Zoom in 2x, Greyscale, and add a white border.
    """
    # 1. Resize (Zoom in 2x)
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # 2. Greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Add White Border (Padding) 
    # This prevents the first/last letter from touching the edge
    return cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)

def process_image(model, reader, filepath):
    img = cv2.imread(filepath)
    if img is None: return

    # YOLO Detect
    results = model.predict(source=img, conf=0.25, verbose=False)
    
    found_plate = False
    for result in results:
        for box in result.boxes:
            found_plate = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop Plate
            plate_crop = img[y1:y2, x1:x2]
            
            # Clean Plate
            clean_plate = preprocess_for_ocr(plate_crop)

            # Read Plate
            ocr_result = reader.readtext(
                clean_plate, 
                allowlist=ALLOWED_CHARS,
                paragraph=False # Keep this False so it sees spaces
            )

            # --- STITCHING LOGIC (The Secret Sauce) ---
            # 1. Filter out low confidence junk (< 0.4)
            valid_parts = [res for res in ocr_result if res[2] > 0.4]

            # 2. Sort parts from Left to Right (by X coordinate)
            # This ensures "B" then "MW" then "123" come in order
            valid_parts.sort(key=lambda x: x[0][0][0])

            # 3. Join them together
            detected_text = "".join([part[1] for part in valid_parts])
            
            # 4. Clean (Remove anything that isn't A-Z or 0-9)
            detected_text = re.sub(r'[^A-Z0-9]', '', detected_text)

            # Length Check: 
            # DK = 7 chars, PL = 7-8 chars, DE = 5-8 chars
            if 5 <= len(detected_text) <= 9:
                print(f"✅ PLATE: {detected_text}")
                
                # Draw
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, detected_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Show the cropped plate to see what OCR saw
                cv2.imshow("Plate Crop", clean_plate)
            else:
                print(f"⚠️  Skipped (Text: {detected_text})")

    if found_plate:
        # Resize large images for display
        if img.shape[1] > 1000:
            scale = 1000 / img.shape[1]
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        cv2.imshow("Live Feed", img)
        cv2.waitKey(2000)
    else:
        print("No plates found.")

def main():
    model, reader = initialize_model()
    
    # Load existing files so we don't re-process them
    for f in os.listdir(WATCH_FOLDER):
        processed_files[f] = get_file_timestamp(os.path.join(WATCH_FOLDER, f))

    print(f"Watching: {WATCH_FOLDER}")

    while True:
        try:
            for filename in os.listdir(WATCH_FOLDER):
                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue
                
                full_path = os.path.join(WATCH_FOLDER, filename)
                t_stamp = get_file_timestamp(full_path)
                
                if filename not in processed_files or t_stamp != processed_files[filename]:
                    print(f"\n📥 NEW: {filename}")
                    if wait_for_write_finish(full_path):
                        processed_files[filename] = t_stamp
                        process_image(model, reader, full_path)

            if cv2.waitKey(10) & 0xFF == ord('q'): break
            time.sleep(0.5)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()