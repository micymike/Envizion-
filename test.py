import cv2
import dlib
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Load Dlib's facial landmark detector
predictor_path = 'shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Load the human face image (replace "neutralface.jpeg" with your actual image path)
face_image = cv2.imread('neutralface.jpeg')
face_image = cv2.resize(face_image, (500, 500))  # Resize for consistency with avatar size

# Define the Tkinter window and canvas
window = tk.Tk()
window.title("Real-Time Avatar Mimicking")

canvas = tk.Label(window)
canvas.pack()

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)

def get_facial_landmarks(gray, rect):
    shape = predictor(gray, rect)
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

def calculate_head_tilt(landmarks):
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    return angle

def is_mouth_open(landmarks, threshold=20):
    upper_lip = landmarks[62]
    lower_lip = landmarks[66]
    distance = np.linalg.norm(upper_lip - lower_lip)
    return distance > threshold

def update_avatar(landmarks, avatar, head_tilt, mouth_open):
    avatar_landmarks = landmarks.copy()

    # Scale and center the landmarks to fit the avatar
    avatar_landmarks -= np.min(avatar_landmarks, axis=0)
    avatar_landmarks = avatar_landmarks / np.max(avatar_landmarks) * (avatar.shape[0] * 0.8)
    avatar_landmarks += (avatar.shape[0] * 0.1, avatar.shape[1] * 0.1)

    avatar_copy = avatar.copy()

    # Rotate the avatar to match head tilt
    center = tuple(np.mean(avatar_landmarks, axis=0).astype(int))
    M = cv2.getRotationMatrix2D(center, head_tilt, 1)
    avatar_copy = cv2.warpAffine(avatar_copy, M, (avatar_copy.shape[1], avatar_copy.shape[0]))

    # Draw eyes on the avatar based on landmarks
    left_eye = avatar_landmarks[36:42].mean(axis=0).astype(int)
    right_eye = avatar_landmarks[42:48].mean(axis=0).astype(int)
    cv2.circle(avatar_copy, tuple(left_eye), 5, (0, 0, 0), -1)
    cv2.circle(avatar_copy, tuple(right_eye), 5, (0, 0, 0), -1)

    # Draw mouth on the avatar
    mouth_points = avatar_landmarks[48:60].astype(int)
    if mouth_open:
        cv2.fillPoly(avatar_copy, [mouth_points], (0, 0, 0))
    else:
        cv2.polylines(avatar_copy, [mouth_points], True, (0, 0, 0), 2)

    # Draw nose on the avatar
    nose_bridge = avatar_landmarks[27:31].astype(int)
    cv2.polylines(avatar_copy, [nose_bridge], False, (0, 0, 0), 2)

    return avatar_copy

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        landmarks = get_facial_landmarks(gray, faces[0])
        head_tilt = calculate_head_tilt(landmarks)
        mouth_open = is_mouth_open(landmarks)
        updated_avatar = update_avatar(landmarks, face_image, head_tilt, mouth_open)
    else:
        updated_avatar = face_image.copy()

    # Convert the updated avatar to RGB for Tkinter display
    avatar_rgb = cv2.cvtColor(updated_avatar, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(avatar_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    canvas.imgtk = img_tk
    canvas.configure(image=img_tk)

    window.after(10, update_frame)

# Start the update loop
update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()