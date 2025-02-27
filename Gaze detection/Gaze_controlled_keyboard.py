import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import time
import pygame
import os

def submit():
    selected_sensitivity = sensitivity_var.get()
    if selected_sensitivity:
        messagebox.showinfo("Selection", f"You selected: {selected_sensitivity}")
        root.destroy()
    else:
        messagebox.showwarning("Warning", "Please select a sensitivity level")
# Create main window
root = tk.Tk()
root.title("Sensitivity Selection Wizard")
root.geometry("300x200")

tk.Label(root, text="Select left-right sensitivity level:", font=("Arial", 12)).pack(pady=10)

sensitivity_var = tk.StringVar(value="")

tk.Radiobutton(root, text="High", variable=sensitivity_var, value="High").pack(anchor="w", padx=20)
tk.Radiobutton(root, text="Medium (Recommended)", variable=sensitivity_var, value="Medium").pack(anchor="w", padx=20)
tk.Radiobutton(root, text="Low", variable=sensitivity_var, value="Low").pack(anchor="w", padx=20)

tk.Button(root, text="Submit", command=submit).pack(pady=20)

root.mainloop()

if sensitivity_var.get() == "High":
    THRESHOLD_HORIZONTAL = 3
elif sensitivity_var.get() == "Medium":
    THRESHOLD_HORIZONTAL = 2
elif sensitivity_var.get() == "Low":
    THRESHOLD_HORIZONTAL = 1

def submit_value():
    try:
        selected_sensitivity = float(sensitivity_var.get())
        messagebox.showinfo("Selection", f"You selected sensitivity: {selected_sensitivity}")
        root.destroy()
    except ValueError:
        messagebox.showwarning("Warning", "Please enter a valid numerical value")

# Create main window
root = tk.Tk()
root.title("Sensitivity Input Wizard")
root.geometry("300x200")

tk.Label(root, text="Enter blink sensitivity percentage (numeric):\n(Recommended = 10)", font=("Arial", 12)).pack(pady=10)

sensitivity_var = tk.StringVar()

entry = tk.Entry(root, textvariable=sensitivity_var)
entry.pack(pady=5)

tk.Button(root, text="Submit", command=submit_value).pack(pady=20)

root.mainloop()

BLINK_SENSITIVITY = int(sensitivity_var.get())/100
    
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
RIGHT_EYEBROW = [300]  # Approximate top of right eyebow
NOSE_TIP = [1]  # Nose tip landmark
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
LEFT_EYE_INDICES = [133, 158, 160, 33, 144, 153]
BLINK_TOP = 159
BLINK_BOTTOM = 145
IRIS1_LEFT = 362
IRIS1_RIGHT = 263
IRIS2_LEFT = 33
IRIS2_RIGHT = 133
THRESHOLD_VERTICAL_UP = 0.1  # Iris closer to top 30%
THRESHOLD_VERTICAL_DOWN = 2 # Iris closer to bottom 70%
SCALE_VERTICAL = 2.2  # Scale factor for vertical sensitivity

# Keyboard setup

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
root.geometry("650x350")

input_field = tk.Entry(root, font=("Arial", 18), width=40)
input_field.grid(row=0, column=0, columnspan=10, pady=10)

keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    ["k", "l", "m", "n", "o", "p", "q", "r", "s"],
    ["t", "u", "v", "w", "x", "y", "z"],
]

# Dictionary to store button references for highlighting
key_buttons = []
for row, key_row in enumerate(keys):
    button_row = []
    for col, key in enumerate(key_row):
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

    if(row == 3):
        # Add space and backspace to the button grid for navigation
        btn = tk.Button(
            root,
            text="__",
            font=("Arial", 14),
            width=4,
            height=2,
            bg="lightgray",
            command=lambda: insert_char(" "),
        )
        btn.grid(row=row + 1, column=7, padx=5, pady=5)
        button_row.append(btn)
        btn = tk.Button(
            root,
            text="⌫",
            font=("Arial", 14),
            width=4,
            height=2,
            bg="lightgray",
            command=backspace,
        )
        btn.grid(row=row + 1, column=8, padx=5, pady=5)
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


# Eye tracker function
def run_eye_tracker():

    #Preset module for vertical and blink threshold

    global current_row, current_col
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
        # Get eye boundaries
        left_eye_top = landmarks[386][1]
        left_eye_bottom = landmarks[374][1]
        eye_height = left_eye_bottom - left_eye_top
        blink_threshold = eye_height * BLINK_SENSITIVITY
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
                # Calculate vertical difference
                left_iris_center_y = np.mean([landmarks[474][1], landmarks[475][1], landmarks[476][1], landmarks[477][1]])
                vertical_ratio = (left_iris_center_y - left_eye_top) / eye_height
                vertical_ratio_scaled = (vertical_ratio - 0.5) * SCALE_VERTICAL  # Center at 0
                # print(vertical_ratio_scaled)

                left_iris_y = np.mean([landmarks[159][1], landmarks[145][1]])

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

                elif vertical_ratio_scaled < -THRESHOLD_VERTICAL_UP and time.time() - previous_time > 1:
                    previous_time = time.time()
                    counter = 0
                    print("Up")
                    if current_row > 0:
                        root.after(0, move_highlight, current_row - 1, current_col)
                        
                elif vertical_ratio_scaled > THRESHOLD_VERTICAL_DOWN and time.time() - previous_time > 1:
                    previous_time = time.time()
                    counter = 0
                    print("Down")
                    if current_row < len(key_buttons) - 1 and current_col < len(key_buttons[current_row + 1]):
                        root.after(0, move_highlight, current_row + 1, current_col)

        cv2.imshow("Gaze Detector", frame)
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
