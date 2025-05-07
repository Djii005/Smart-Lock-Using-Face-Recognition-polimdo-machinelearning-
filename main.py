#!/usr/bin/env python3
import cv2
import os
import numpy as np
import time
from datetime import datetime
import threading
import pickle
from flask import Flask, render_template, Response, request, jsonify

# Configuration
DATA_DIR = "face_data"
USERS_DIR = "authorized_users"
MODEL_FILE = "lbph_model.yml"
LABELS_FILE = "face_labels.pickle"
CASCADE_FILE = r'C:\Users\Adji\AppData\Roaming\Python\Python313\site-packages\cv2\data\haarcascade_frontalface_default.xml'
CAMERA_RESOLUTION = (320, 240)
PORT = 8080

# Create directories if they don't exist
for directory in [DATA_DIR, USERS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create Flask app
app = Flask(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
if face_cascade.empty():
    raise ValueError(f"Error loading Haar cascade from {CASCADE_FILE}")

# Initialize face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
face_names = {}  # Dictionary to store label->name mappings
model_loaded = False
last_recognized_name = "Unknown"
last_confidence = 0

def load_face_model():
    """Load the face recognition model if it exists"""
    global model_loaded, face_names
    
    try:
        # Load face recognizer model
        face_recognizer.read(MODEL_FILE)
        
        # Load labels
        if os.path.exists(LABELS_FILE):
            with open(LABELS_FILE, 'rb') as f:
                face_names = pickle.load(f)
        
        # Also load from users directory
        for user_file in os.listdir(USERS_DIR):
            if os.path.isfile(os.path.join(USERS_DIR, user_file)):
                parts = user_file.split('-', 1)
                if len(parts) == 2:
                    user_id = int(parts[0])
                    user_name = parts[1]
                    face_names[user_id] = user_name
        
        model_loaded = True
        print("Face recognition model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading face model: {e}")
        model_loaded = False
        return False

def detect_and_recognize_faces(frame):
    """Detect and recognize faces in the frame"""
    global last_recognized_name, last_confidence
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Check if any faces were detected
    if len(faces) == 0:
        last_recognized_name = "Unknown"
        last_confidence = 0
    
    # Process each face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Draw rectangle around the face
        color = (0, 0, 255)  # Default: red (unknown)
        name = "Unknown"
        confidence = 0
        
        # Recognize the face if model is loaded
        if model_loaded:
            try:
                label, confidence_score = face_recognizer.predict(face_roi)
                confidence = 100 - confidence_score  # Convert to percentage
                
                if confidence >= 60:  # Confidence threshold
                    if label in face_names:
                        name = face_names[label]
                        color = (0, 255, 0)  # Green (recognized)
                        last_recognized_name = name
                        last_confidence = confidence
            except Exception as e:
                print(f"Recognition error: {e}")
        
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Display the name and confidence
        text = f"{name} ({confidence:.1f}%)" if confidence > 0 else name
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def camera_stream():
    """Generator function for the camera stream"""
    global output_frame, camera
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    
    # Allow camera to warm up
    time.sleep(2.0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        # Process the frame
        processed_frame = detect_and_recognize_faces(frame)
        
        # Acquire the lock
        with lock:
            output_frame = processed_frame.copy()
            
        # Small delay to reduce CPU usage
        time.sleep(0.03)
    
    camera.release()

# Variables for face capture
capture_in_progress = False
capture_count = 0
capture_max = 50
capture_user_id = 0
capture_user_name = ""
capture_completed = False

def start_face_capture(user_name):
    """Prepare for face capture"""
    global capture_in_progress, capture_count, capture_user_id
    global capture_user_name, capture_completed
    
    # Reset capture variables
    capture_count = 0
    capture_completed = False
    capture_user_name = user_name
    
    # Find next user ID
    capture_user_id = 1
    if os.path.exists(USERS_DIR):
        existing_ids = [int(f.split('-')[0]) for f in os.listdir(USERS_DIR) 
                        if os.path.isfile(os.path.join(USERS_DIR, f)) and f.split('-')[0].isdigit()]
        if existing_ids:
            capture_user_id = max(existing_ids) + 1
    
    # Create user directory
    user_dir = os.path.join(DATA_DIR, f"{capture_user_id}")
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    # Start capture
    capture_in_progress = True
    
    # Start face capture in a separate thread
    threading.Thread(target=capture_face_images, daemon=True).start()
    
    return True

def capture_face_images():
    """Capture face images in a background thread"""
    global capture_in_progress, capture_count, capture_completed
    global output_frame, camera
    
    try:
        while capture_in_progress and capture_count < capture_max:
            # Get current frame
            with lock:
                if output_frame is None:
                    continue
                frame = output_frame.copy()
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) == 1:  # Only capture if exactly one face is detected
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Save the face image
                file_name = os.path.join(DATA_DIR, f"{capture_user_id}", f"face_{capture_count}.jpg")
                cv2.imwrite(file_name, face_roi)
                capture_count += 1
                
                # Small delay between captures
                time.sleep(0.2)
            
            # Larger delay if no valid face
            else:
                time.sleep(0.1)
    
    except Exception as e:
        print(f"Error in face capture: {e}")
    
    finally:
        # Create a user file
        with open(os.path.join(USERS_DIR, f"{capture_user_id}-{capture_user_name}"), "w") as f:
            f.write(f"User ID: {capture_user_id}\nUser Name: {capture_user_name}\nSamples: {capture_count}\n")
        
        # Update capture status
        capture_in_progress = False
        capture_completed = True
        print(f"Captured {capture_count} images for user {capture_user_name}")

def train_face_recognizer():
    """Train the face recognizer with the collected data"""
    global model_loaded, face_names
    
    try:
        faces = []
        labels = []
        face_names = {}
        
        # Check if we have any user data
        if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
            return (False, "No user data found. Please add users first.", 0, 0)
        
        # Process each user's data
        for user_id in os.listdir(DATA_DIR):
            user_dir = os.path.join(DATA_DIR, user_id)
            if not os.path.isdir(user_dir):
                continue
            
            user_id_int = int(user_id)
            
            # Find user name
            user_name = f"User-{user_id}"
            for user_file in os.listdir(USERS_DIR):
                if user_file.startswith(f"{user_id}-"):
                    user_name = user_file.split('-', 1)[1]
                    break
            
            # Add to face names dictionary
            face_names[user_id_int] = user_name
            
            # Process each face image
            for image_file in os.listdir(user_dir):
                if not image_file.endswith('.jpg'):
                    continue
                    
                image_path = os.path.join(user_dir, image_file)
                face_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if face_img is not None:
                    faces.append(face_img)
                    labels.append(user_id_int)
        
        if not faces:
            return (False, "No face data found", 0, 0)
        
        # Save the face names dictionary
        with open(LABELS_FILE, 'wb') as f:
            pickle.dump(face_names, f)
        
        # Train the model
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save(MODEL_FILE)
        
        # Update model status
        model_loaded = True
        
        return (True, "Training completed successfully", len(faces), len(set(labels)))
    
    except Exception as e:
        print(f"Error training model: {e}")
        return (False, str(e), 0, 0)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

def generate():
    """Video streaming generator function"""
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
                
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition_status')
def recognition_status():
    """Return the current recognition status"""
    global last_recognized_name, last_confidence
    
    # Return empty if no face recognized
    if last_recognized_name == "Unknown" or last_confidence < 1:
        return jsonify({"name": "", "confidence": 0})
    
    return jsonify({"name": last_recognized_name, "confidence": last_confidence})

@app.route('/start_capture')
def start_capture():
    """Start face capture for a new user"""
    name = request.args.get('name', '')
    
    if not name:
        return jsonify({"success": False, "message": "Name is required"})
    
    if capture_in_progress:
        return jsonify({"success": False, "message": "Capture already in progress"})
    
    success = start_face_capture(name)
    return jsonify({"success": success})

@app.route('/capture_progress')
def capture_progress():
    """Return the current capture progress"""
    global capture_count, capture_completed
    
    return jsonify({
        "count": capture_count,
        "max": capture_max,
        "completed": capture_completed
    })

@app.route('/train_model')
def train_model():
    """Train the face recognition model"""
    if capture_in_progress:
        return jsonify({"success": False, "message": "Cannot train while capture is in progress"})
    
    success, message, face_count, user_count = train_face_recognizer()
    
    return jsonify({
        "success": success,
        "message": message,
        "faces": face_count,
        "users": user_count
    })

if __name__ == '__main__':
    # Load face model if exists
    load_face_model()
    
    # Start the camera thread
    t = threading.Thread(target=camera_stream, daemon=True)
    t.start()
    
    # Start the Flask web server
    print(f"Starting web server on port {PORT}")
    print(f"Open http://[Your Raspberry Pi IP]:{PORT} in a web browser")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)