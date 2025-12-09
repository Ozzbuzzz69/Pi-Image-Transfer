"""
===============================================================================
FILE: Ultrasound.py
TYPE: Client (Run this on the Raspberry Pi)

DESCRIPTION: 
    1. Monitors distance using an HC-SR04 Ultrasonic sensor.
    2. Detects if an object (car) is within range (< 50cm).
    3. Triggers a high-speed camera burst (10 photos).
    4. Sends photos over TCP to the server immediately.

HARDWARE:
    - GPIO 23: Trigger (Output)
    - GPIO 24: Echo (Input) - Needs Voltage Divider!
    - Pi Camera Module
===============================================================================
"""

import RPi.GPIO as GPIO
import time
import socket
import struct
import io
from picamera2 import Picamera2

# ==========================================
# CONFIGURATION
# ==========================================

# --- GPIO PINS ---
TRIG = 23
ECHO = 24

# --- DETECTION LOGIC ---
DETECTION_DISTANCE = 50   # Trigger if object is closer than X cm
RESET_DISTANCE = 65       # Reset trigger only after object is further than X cm

# --- NETWORK SETTINGS ---
# IMPORTANT: Change this to your PC's IP address!
SERVER_IP = '192.168.14.204' 
SERVER_PORT = 8000

# --- CAMERA SETTINGS ---
TOTAL_IMAGES = 5         # How many photos to take per car
INTERVAL = 0.2            # Time between photos (seconds)
SHUTTER_SPEED = 4000      # Microseconds (2500us = 1/400th sec) - Freezes motion
ANALOG_GAIN = 6.0         # Brightness gain (High ISO)

# ==========================================
# HARDWARE INITIALIZATION
# ==========================================

# Setup GPIO Mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Initialize Camera GLOBALLY
# We do this OUTSIDE the loop so the camera is always "warm" and ready.
# If we did this inside the loop, we would miss the car while waiting for startup.
print("[*] Initializing Camera (Please wait)...")
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1920, 1080)})
picam2.configure(config)
picam2.start() # Start the viewfinder/sensor

# Lock camera settings for consistency
picam2.set_controls({
    "ExposureTime": SHUTTER_SPEED,
    "AnalogueGain": ANALOG_GAIN,
    "AeEnable": False # Disable Auto-Exposure (Manual Mode)
})
print("[*] Camera Ready.")

# ==========================================
# FUNCTIONS
# ==========================================

def get_distance():
    """
    Sends a ping and measures the echo time.
    Includes a SAFETY TIMEOUT to prevent the code from hanging forever
    if the sensor disconnects or misses a pulse.
    
    Returns:
        float: Distance in cm
        None: If measurement timed out
    """
    # 1. Send 10 microsecond pulse to TRIG
    GPIO.output(TRIG, False)
    time.sleep(0.02) # Let signal settle
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Define a max wait time (0.04s = approx 6-7 meters max range)
    timeout = time.time() + 0.04 

    # 2. Wait for Echo Start (Signal goes High)
    pulse_start = time.time()
    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return None # Timed out waiting for signal
        pulse_start = time.time()

    # 3. Wait for Echo End (Signal goes Low)
    pulse_end = time.time()
    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return None # Timed out waiting for signal end
        pulse_end = time.time()

    # 4. Calculate Distance
    # Distance = (Time * Speed of Sound (34300 cm/s)) / 2
    duration = pulse_end - pulse_start
    distance = duration * 17150
    return round(distance, 2)

def send_burst():
    """
    Connects to the server, captures 10 images to RAM, and uploads them.
    """
    print(f"[*] Connecting to {SERVER_IP}...")
    
    try:
        # Create socket connection
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.settimeout(5) # Give up if server doesn't reply in 5s
            client_socket.connect((SERVER_IP, SERVER_PORT))
            
            print("[*] Connection successful. Sending burst...")
            
            # 'io.BytesIO' acts like a file but lives in RAM (Memory).
            # This is much faster than saving to the SD card.
            stream = io.BytesIO()

            for i in range(1, TOTAL_IMAGES + 1):
                loop_start = time.time()

                # Clear the stream for the new photo
                stream.seek(0)
                stream.truncate()
                
                # Capture Photo
                picam2.capture_file(stream, format='jpeg')
                
                # Prepare Data
                image_data = stream.getvalue()
                file_size = len(image_data)

                # PROTOCOL: Send Size (4 bytes) -> Then Send Image
                client_socket.sendall(struct.pack('>I', file_size))
                client_socket.sendall(image_data)
                
                print(f"    - Sent Image {i}/{TOTAL_IMAGES} ({file_size} bytes)")

                # Wait slightly to match the desired INTERVAL
                elapsed = time.time() - loop_start
                time.sleep(max(0, INTERVAL - elapsed))

    except (ConnectionRefusedError, socket.timeout):
        print(f"[!] Server {SERVER_IP} is unreachable. Is it running?")
    except Exception as e:
        print(f"[!] Error during upload: {e}")

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

# State Flag: keeps track if we already detected the current car
car_present = False

print("[*] System Running... (Press Ctrl+C to stop)")

try:
    while True:
        dist = get_distance()
        
        # If sensor timed out, skip this loop iteration
        if dist is None:
            continue

        # Print distance on the same line (using '\r')
        print(f"Distance: {dist} cm   ", end='\r') 

        # --- LOGIC 1: DETECT CAR ---
        # If object is close AND we haven't flagged it yet
        if dist <= DETECTION_DISTANCE and not car_present:
            print(f"\n🚗 Car detected at {dist}cm! Starting capture...")
            car_present = True # Set flag so we don't trigger again for same car
            
            send_burst() # Call the camera/network function
            
            print("[*] Burst complete. Waiting for area to clear...")

        # --- LOGIC 2: RESET SYSTEM ---
        # If object is far away AND the flag is still True
        elif dist > RESET_DISTANCE and car_present:
            print("\n✅ Path clear. System reset and ready.")
            car_present = False # Reset flag, ready for next car

        # Small delay to reduce CPU usage
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n[!] Stopping System...")
finally:
    # Cleanup hardware resources on exit
    picam2.stop()
    GPIO.cleanup()
    print("[*] Cleanup complete. Goodbye.")