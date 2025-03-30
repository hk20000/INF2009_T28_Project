import cv2
import time
import pickle
import numpy as np
import face_recognition
import mediapipe as mp
import sqlite3
import threading

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect('engagement.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS engagement (
                 speaker_id INTEGER, 
                 timestamp TEXT, 
                 expression TEXT,
                 transcription TEXT)''')
    conn.commit()
    return conn, c

conn, c = init_db()

# Load pre-trained face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize USB webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam
cap.set(3, 1040)  # Set width
cap.set(4, 880)  # Set height

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

cv_scaler = 3  # Scaling factor to improve performance

face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define landmarks for eyes and mouth
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14, 312, 82]

# Function to detect facial expressions and states
def detect_expression(frame):
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get eye landmarks
            left_eye = [(int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)) for i in RIGHT_EYE]
            
            # Get mouth landmarks
            mouth = [(int(face_landmarks.landmark[i].x * width), int(face_landmarks.landmark[i].y * height)) for i in MOUTH]
            
            # Calculate eye aspect ratio
            left_eye_height = abs(left_eye[1][1] - left_eye[5][1]) + abs(left_eye[2][1] - left_eye[4][1])
            right_eye_height = abs(right_eye[1][1] - right_eye[5][1]) + abs(right_eye[2][1] - right_eye[4][1])
            eye_width = abs(left_eye[0][0] - left_eye[3][0])
            
            eye_aspect_ratio = (left_eye_height + right_eye_height) / (2.0 * eye_width)
            eye_status = "Open" if eye_aspect_ratio > 0.2 else "Closed"
            
            # Calculate mouth aspect ratio
            mouth_height = abs(mouth[2][1] - mouth[3][1])
            mouth_width = abs(mouth[0][0] - mouth[1][0])
            mouth_aspect_ratio = mouth_height / mouth_width
            mouth_status = "Open" if mouth_aspect_ratio > 0.3 else "Closed"

            # Classify state
            if eye_status == "Closed" and mouth_status == "Closed":
                state = "Sleeping"
                color = (0, 0, 255)  # Red
            elif eye_status == "Open" and mouth_status == "Closed":
                state = "Engaged"
                color = (0, 255, 0)  # Green
            else:
                state = "Talking"
                color = (255, 255, 0)  # Yellow

            return state, color

    return "No Face", (255, 255, 255)

# Process frame for face recognition and facial expression detection
def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all faces and their encodings
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    
    return frame

# Draw results on the frame, including detected faces and states
def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale face locations back up
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Detect facial state
        state, state_color = detect_expression(frame)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), state_color, 3)
        
        # Draw a label below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), state_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{name} - {state}", (left + 6, top - 6), font, 0.5, (255, 255, 255), 1)
        
        # Insert state into the database
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        c.execute('INSERT INTO engagement (timestamp, expression, speaker_id, transcription) VALUES (?, ?, ?, ?)', 
                  (timestamp, state, name, None))
        conn.commit()
    
    return frame

# Calculate FPS for performance tracking
def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Main loop to handle video stream
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    processed_frame = process_frame(frame)
    display_frame = draw_results(processed_frame)
    
    # Show FPS
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display video feed
    cv2.imshow('Video', display_frame)
    
    # Quit on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
conn.close()
