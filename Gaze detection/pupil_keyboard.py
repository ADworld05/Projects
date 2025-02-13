import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import time
import pygame
import os

# Eye tracking setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

# Landmark indices for eyes and iris
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_TOP_BOTTOM = [386, 374]  # Top and bottom of the left eye
RIGHT_EYE_TOP_BOTTOM = [159, 145]  # Top and bottom of the right eye
LEFT_EYEBROW = [70]  # Approximate top of left eyebrow
RIGHT_EYEBROW = [300]  # Approximate top of right eyebrow
NOSE_TIP = [1]  # Nose tip landmark
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_EYE_INDICES = [133, 158, 160, 33, 144, 153]
BLINK_TOP = 159
BLINK_BOTTOM = 145
IRIS1_LEFT = 362
IRIS1_RIGHT = 263
IRIS2_LEFT = 33
IRIS2_RIGHT = 133
THRESHOLD_HORIZONTAL = 2
THRESHOLD_VERTICAL_UP = 0.3  # Iris closer to top 30%
THRESHOLD_VERTICAL_DOWN = 0.9 # Iris closer to bottom 70%
SCALE_VERTICAL = 2.2  # Scale factor for vertical sensitivity

# Initialize pygame for sound
pygame.mixer.init()

def play_sound(char):
    """Plays sound for the selected key."""
    try:
        sound_file = f"sounds/{char}.mp3"
        if os.path.exists(sound_file):  # Ensure the file exists before playing
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        else:
            print(f"Sound file not found: {sound_file}")
    except Exception as e:
        print(f"Error playing sound for {char}: {e}")

def insert_char(char):
    """Inserts the clicked character into the input field."""
    current_text = input_field.get()
    input_field.delete(0, tk.END)
    input_field.insert(0, current_text + char)

def backspace():
    """Deletes the last character in the input field."""
    current_text = input_field.get()
    input_field.delete(0, tk.END)
    input_field.insert(0, current_text[:-1])

root = tk.Tk()
root.title("Iris-Controlled Virtual Keyboard")
root.geometry("650x450")

input_field = tk.Entry(root, font=("Arial", 18), width=40)
input_field.grid(row=0, column=0, columnspan=10, pady=10)

keys = [
    ["a", "b", "c", "d", "e"],  # Row 1
    ["f", "g", "h", "i", "j"],  # Row 2
    ["k", "l", "m", "n", "o", "__"],  # Row 3 with space key
    ["p", "q", "r", "s", "t", "⌫"],  # Row 4 with backspace key
    ["u", "v", "w", "x", "y", "z"],  # Row 5
]

# Dictionary to store button references for highlighting
key_buttons = []
for row, key_row in enumerate(keys):
    button_row = []
    for col, key in enumerate(key_row):
        if key == "__":
            btn = tk.Button(
                root,
                text="__",
                font=("Arial", 14),
                width=4,
                height=2,
                bg="lightgray",
                command=lambda char=" ": insert_char(char),
            )
        elif key == "⌫":
            btn = tk.Button(
                root,
                text="⌫",
                font=("Arial", 14),
                width=4,
                height=2,
                bg="lightgray",
                command=backspace,
            )
        else:
            btn = tk.Button(
                root,
                text=key,
                font=("Arial", 14),
                width=4,
                height=2,
                bg="lightgray",
                command=lambda char=key: insert_char(char),
            )
        btn.grid(row=row + 1, column=col, padx=5, pady=5)
        button_row.append(btn)
    key_buttons.append(button_row)

current_row, current_col = 2, 4
key_buttons[current_row][current_col].config(bg="yellow")

def get_eye_region(landmarks, indices):
    """Extract eye region landmarks."""
    return np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in indices], dtype=np.int32)

def move_highlight(new_row, new_col):
    global current_row, current_col
    key_buttons[current_row][current_col].config(bg="lightgray")
    current_row, current_col = new_row, new_col
    key_buttons[current_row][current_col].config(bg="yellow")

    # Get the character at the new position and play the sound
    char = key_buttons[current_row][current_col].cget("text")
    if char == "__":
        char = "space"
    elif char == "⌫":
        char = "backspace"
    
    play_sound(char)

def blink_action():
    key_buttons[current_row][current_col].invoke()

def run_eye_tracker():
    global current_row, current_col
    cap = cv2.VideoCapture(0)
    
    # Wait until face is detected
    face_detected = False
    while not face_detected:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_detected = True
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
                # Get eye boundaries
                left_eye_top = landmarks[386][1]
                left_eye_bottom = landmarks[374][1]
                eye_height = left_eye_bottom - left_eye_top
                blink_threshold = eye_height * 0.2
            else:
                print("Please position face in camera view...")
                time.sleep(0.5)

    previous_time = time.time()
    counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]
                h, w, _ = frame.shape

                def get_landmark_coords(landmark_indices):
                    return np.array([
                        (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                        for i in landmark_indices
                    ])
                
                left_iris = get_landmark_coords(LEFT_IRIS)
                right_iris = get_landmark_coords(RIGHT_IRIS)
                left_eye = get_landmark_coords(LEFT_EYE_TOP_BOTTOM)
                right_eye = get_landmark_coords(RIGHT_EYE_TOP_BOTTOM)
                left_eyebrow = get_landmark_coords(LEFT_EYEBROW)
                right_eyebrow = get_landmark_coords(RIGHT_EYEBROW)
                nose_tip = get_landmark_coords(NOSE_TIP)
            
                # Calculate iris center (average of iris points)
                left_iris_center = np.mean(left_iris, axis=0).astype(int)
                right_iris_center = np.mean(right_iris, axis=0).astype(int)
            
                # Calculate vertical difference
                left_iris_center_y = np.mean([landmarks[474][1], landmarks[475][1], landmarks[476][1], landmarks[477][1]])
                vertical_ratio = (left_iris_center_y - left_eye_top) / eye_height
                vertical_ratio_scaled = (vertical_ratio - 0.5) * SCALE_VERTICAL  # Center at 0

                iris_x1 = landmarks[473][0]
                left_x1 = landmarks[IRIS1_LEFT][0]
                right_x1 = landmarks[IRIS1_RIGHT][0]
                left_diff1 = abs(iris_x1 - left_x1)
                right_diff1 = abs(iris_x1 - right_x1)
                ratio1 = left_diff1 / right_diff1

                iris_x2 = landmarks[468][0]
                left_x2 = landmarks[IRIS2_LEFT][0]
                right_x2 = landmarks[IRIS2_RIGHT][0]
                left_diff2 = abs(iris_x2 - left_x2)
                right_diff2 = abs(iris_x2 - right_x2)
                ratio2 = right_diff2 / left_diff2

                left_eye = get_eye_region(landmarks, LEFT_EYE_INDICES)
                right_eye = get_eye_region(landmarks, RIGHT_EYE_INDICES)
                top_y = landmarks[BLINK_TOP][1]
                bottom_y = landmarks[BLINK_BOTTOM][1]
                blink_diff = bottom_y - top_y

                if blink_diff < blink_threshold and time.time() - previous_time > 0.8:
                    previous_time = time.time()
                    counter += 1
                    if counter >= 2:
                        root.after(0, blink_action)
                        print("Blink")
                        counter = 0

                if ratio2 > THRESHOLD_HORIZONTAL and time.time() - previous_time > 1:
                    counter = 0                    
                    previous_time = time.time()
                    print("Left")
                    if current_col > 0:
                        root.after(0, move_highlight, current_row, current_col - 1)
                        print("Left")
                        
                elif ratio1 > THRESHOLD_HORIZONTAL and time.time() - previous_time > 1:
                    counter = 0
                    previous_time = time.time()
                    print("Right")
                    if current_col < len(key_buttons[current_row]) - 1:
                        root.after(0, move_highlight, current_row, current_col + 1)
                        print("Right")

                elif vertical_ratio_scaled < -THRESHOLD_VERTICAL_UP and time.time() - previous_time > 1.5:
                    previous_time = time.time()
                    counter = 0
                    print("Up")
                    if current_row > 0:
                        root.after(0, move_highlight, current_row - 1, current_col)
                        
                elif vertical_ratio_scaled > THRESHOLD_VERTICAL_DOWN and time.time() - previous_time > 1.5:
                    previous_time = time.time()
                    counter = 0
                    print("Down")
                    if current_row < len(key_buttons) - 1 and current_col < len(key_buttons[current_row + 1]):
                        root.after(0, move_highlight, current_row + 1, current_col)

                # Visualization
                cv2.circle(frame, (int(face_landmarks.landmark[473].x * frame.shape[1]), int(face_landmarks.landmark[473].y * frame.shape[0])), 3, (0, 255, 255), -1)
                cv2.circle(frame, (int(face_landmarks.landmark[468].x * frame.shape[1]), int(face_landmarks.landmark[468].y * frame.shape[0])), 3, (0, 255, 255), -1)          
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
                cv2.putText(frame, f"Eye controlled keyboard- NIT Durgapur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Vertical Ratio: {vertical_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Blink Difference: {int(blink_diff)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Start the eye tracker in a separate thread
eye_tracker_thread = threading.Thread(target=run_eye_tracker, daemon=True)
eye_tracker_thread.start()

# Start the Tkinter main loop
root.mainloop()