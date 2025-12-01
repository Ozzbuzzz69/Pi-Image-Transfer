"""
===============================================================================
FILE: ImgReciever.py
DESCRIPTION: 
    A lightweight TCP Server designed to receive images from a client 
    (e.g., Raspberry Pi) and save them to a local folder.
    
PROTOCOL:
    It uses a Length-Prefix protocol to ensure complete file transfers:
    [4 Bytes File Size] + [Image Data Payload]
===============================================================================
"""

import socket
import os
import struct
import time

# --- CONFIGURATION ---
SERVER_IP = '0.0.0.0'   # Listen on all network interfaces
SERVER_PORT = 8000      # The port to open
SAVE_FOLDER = 'received_images' # Local folder name

def start_server():
    """
    Initializes the TCP server, waits for connections, and handles
    incoming image data streams.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # Initialize Socket (IPv4, TCP)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # SO_REUSEADDR allows you to restart the script immediately without 
    # waiting for the OS to release the port.
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(5) # Allow a backlog of 5 connections

    print(f"[*] Server listening on {SERVER_IP}:{SERVER_PORT}")
    print(f"[*] Saving images to '{os.path.abspath(SAVE_FOLDER)}'")

    try:
        # OUTER LOOP: Keeps the server running continuously to accept new clients
        while True:
            print("[*] Waiting for a connection...")
            conn, addr = server_socket.accept()
            print(f"[*] Connected by {addr}")

            try:
                # INNER LOOP: Keeps receiving images from the CURRENT client
                image_count = 0
                while True:
                    # 1. Receive the file size header (4 bytes)
                    data_size = conn.recv(4)
                    
                    # If recv returns empty bytes, the client disconnected gracefully
                    if not data_size:
                        print(f"[*] Client {addr} disconnected.")
                        break
                    
                    # Unpack the 4 bytes into an unsigned integer
                    file_size = struct.unpack('>I', data_size)[0]
                    
                    # 2. Receive the actual image data loop
                    received_payload = b""
                    remaining_bytes = file_size
                    
                    while remaining_bytes > 0:
                        # Receive up to 4KB chunks at a time
                        chunk = conn.recv(min(4096, remaining_bytes))
                        if not chunk:
                            break
                        received_payload += chunk
                        remaining_bytes -= len(chunk)

                    # 3. Save the data to disk
                    # Timestamp ensures unique filenames even if count resets
                    timestamp = int(time.time())
                    image_count += 1
                    filename = os.path.join(SAVE_FOLDER, f"img_{timestamp}_{image_count}.jpg")
                    
                    with open(filename, 'wb') as f:
                        f.write(received_payload)
                    
                    print(f"    - Received Image #{image_count} ({file_size} bytes) -> Saved as {filename}")

            except Exception as e:
                print(f"[!] Error handling client {addr}: {e}")
            finally:
                # Close the specific client connection, but keep server running
                conn.close()

    except KeyboardInterrupt:
        print("\n[!] Server stopped manually.")
    finally:
        # Close the main server socket on exit
        server_socket.close()

if __name__ == '__main__':
    start_server()