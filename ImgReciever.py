"""
===============================================================================
FILE: ImgReciever.py
TYPE: Server (Run this on your Computer)

DESCRIPTION: 
    A robust TCP Server designed to receive image streams from a Raspberry Pi.
    It implements a 'Length-Prefix' protocol to ensure images are not corrupted
    during network transfer.

PROTOCOL:
    1. Client sends 4 bytes -> Represents the file size (Integer).
    2. Client sends N bytes -> The actual image data.
    3. Server reads exactly N bytes and saves the file.
===============================================================================
"""

import socket
import os
import struct
import time

# ==========================================
# CONFIGURATION
# ==========================================
# '0.0.0.0' tells the server to accept connections from ANY IP address
# (Wi-Fi, Ethernet, localhost, etc.)
SERVER_IP = '0.0.0.0'   

# Port to listen on. Must match the port in the Client script.
SERVER_PORT = 8000      

# Folder where images will be saved.
SAVE_FOLDER = 'received_images' 

def recv_exact(sock, n_bytes):
    """
    Receives EXACTLY 'n_bytes' from the socket.
    
    Why this is needed:
    TCP streams can be fragmented. If you ask for 50,000 bytes, the network 
    might give you 1,000 bytes now and 49,000 bytes a millisecond later.
    This function loops until it has glued all the pieces together.
    
    Args:
        sock (socket): The active client connection.
        n_bytes (int): The number of bytes we expect to receive.
        
    Returns:
        bytes: The complete data block, or None if connection fails.
    """
    data = b''
    while len(data) < n_bytes:
        # Calculate how many bytes are left to read
        bytes_needed = n_bytes - len(data)
        
        # Read a chunk (up to 4096 bytes at a time, or whatever is left)
        packet = sock.recv(min(4096, bytes_needed))
        
        # If packet is empty, the client disconnected unexpectedly
        if not packet:
            return None
            
        data += packet
    return data

def start_server():
    """
    Main server loop. Initializes the socket and handles incoming files.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    # Context Manager ('with') automatically closes the socket if code crashes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        
        # Allow the script to restart immediately without waiting for port timeout
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind the socket to the IP and Port
        server_socket.bind((SERVER_IP, SERVER_PORT))
        
        # Start listening (allow backlog of 5 connections)
        server_socket.listen(5)
        
        print(f"[*] Server listening on {SERVER_IP}:{SERVER_PORT}")
        print(f"[*] Saving images to: {os.path.abspath(SAVE_FOLDER)}")

        try:
            # OUTER LOOP: Accept new clients (e.g., if Pi reboots)
            while True:
                print("[*] Waiting for connection...")
                conn, addr = server_socket.accept()
                
                with conn: # Auto-close this specific client connection when done
                    print(f"[*] Connected to client: {addr}")
                    image_count = 0
                    
                    # INNER LOOP: Receive multiple images from the same client
                    try:
                        while True:
                            # 1. READ HEADER (4 bytes)
                            # Contains the size of the incoming image
                            raw_msglen = recv_exact(conn, 4)
                            if not raw_msglen:
                                break # Client disconnected normally
                            
                            # Unpack bytes into an integer ('>I' = Big-Endian Unsigned Int)
                            file_size = struct.unpack('>I', raw_msglen)[0]
                            
                            # 2. READ PAYLOAD (Image Data)
                            image_data = recv_exact(conn, file_size)
                            if not image_data:
                                break # Data corrupted or connection lost
                            
                            # 3. SAVE TO DISK
                            timestamp = int(time.time())
                            image_count += 1
                            filename = os.path.join(SAVE_FOLDER, f"img_{timestamp}_{image_count}.jpg")
                            
                            with open(filename, 'wb') as f:
                                f.write(image_data)
                                
                            print(f"    -> Received Image #{image_count} | Size: {file_size} bytes")
                            
                    except Exception as e:
                        print(f"[!] Error processing client {addr}: {e}")
                    
                    print(f"[*] Client {addr} disconnected.")

        except KeyboardInterrupt:
            print("\n[!] Server stopped by user.")

if __name__ == '__main__':
    start_server()