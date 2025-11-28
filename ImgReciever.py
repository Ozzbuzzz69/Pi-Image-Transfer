import socket
import os
import struct
import time

# CONFIGURATION
SERVER_IP = '0.0.0.0'
SERVER_PORT = 8000
SAVE_FOLDER = 'received_images'

def start_server():
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # This allows you to restart the script immediately without "Address already in use" errors
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    server_socket.bind((SERVER_IP, SERVER_PORT))
    server_socket.listen(5) # Allow a backlog of connections

    print(f"[*] Server listening on {SERVER_IP}:{SERVER_PORT}")
    print(f"[*] Saving images to '{os.path.abspath(SAVE_FOLDER)}'")

    try:
        # OUTER LOOP: Keeps the server running to accept new clients
        while True:
            print("[*] Waiting for a connection...")
            conn, addr = server_socket.accept()
            print(f"[*] Connected by {addr}")

            try:
                # INNER LOOP: Keeps receiving images from the current client
                image_count = 0
                while True:
                    # 1. Receive the file size
                    data_size = conn.recv(4)
                    
                    # If recv returns empty bytes, the client disconnected
                    if not data_size:
                        print(f"[*] Client {addr} disconnected.")
                        break
                    
                    file_size = struct.unpack('>I', data_size)[0]
                    
                    # 2. Receive the actual image data
                    received_payload = b""
                    remaining_bytes = file_size
                    while remaining_bytes > 0:
                        chunk = conn.recv(min(4096, remaining_bytes))
                        if not chunk:
                            break
                        received_payload += chunk
                        remaining_bytes -= len(chunk)

                    # 3. Save the data
                    # We add a timestamp to ensure uniqueness across different sessions
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
        server_socket.close()

if __name__ == '__main__':
    start_server()