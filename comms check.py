import cv2
import socket
import pickle
import struct

# 1. Initialize the socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 🔴 REPLACE THIS WITH YOUR EXACT WINDOWS PC ETHERNET IP
PC_ETHERNET_IP = '192.168.23.140'  
PORT = 9999

print(f"⏳ Connecting to Windows AI PC at {PC_ETHERNET_IP}...")
client_socket.connect((PC_ETHERNET_IP, PORT))
print("🚀 Connected successfully! Streaming webcam frames...")

# 2. Open Laptop Webcam
vid = cv2.VideoCapture(0)
# Set to low resolution to keep network speeds lightning fast
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret: 
            break
            
        # Flip frame so it's not mirrored, then compress to JPEG in RAM
        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = pickle.dumps(buffer)
        
        # Pack frame size header and send binary payload
        message = struct.pack("Q", len(data)) + data
        client_socket.sendall(message)
        
except Exception as e:
    print(f"💥 Stream stopped: {e}")

finally:
    vid.release()
    client_socket.close()