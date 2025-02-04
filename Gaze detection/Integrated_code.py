import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import time

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
THRESHOLD_VERTICAL_U = 0.12
THRESHOLD_VERTICAL_D = 0.35
THRESHOLD_HORIZONTAL = 2

# Keyboard setup

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
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    ["z", "x", "c", "v", "b", "n", "m"],
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
        # initial_iris_y = np.mean([landmarks[159][1], landmarks[145][1]])
        top_y = landmarks[BLINK_TOP][1]
        bottom_y = landmarks[BLINK_BOTTOM][1]
        blink_threshold = (bottom_y - top_y) * 0.25
    else:
        # initial_iris_y = None
        blink_threshold = None
        print("ERROR: No face detected !!")

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
                left_iris_diff = abs(left_iris_center[1] - left_eyebrow[0][1]) / abs(left_eyebrow[0][1] - nose_tip[0][1])
                right_iris_diff = abs(right_iris_center[1] - right_eyebrow[0][1]) / abs(right_eyebrow[0][1] - nose_tip[0][1])

                # left_iris_y = np.mean([landmarks[159][1], landmarks[145][1]])

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

                print(int(left_iris_diff*100),int(right_iris_diff*100))

                if blink_diff < blink_threshold and time.time() - previous_time > 0.8:
                    previous_time = time.time()
                    counter += 1
                    if counter >= 2:
                        root.after(0, blink_action)
                        print("Blink")
                        counter = 0

                # vertical_movement = left_iris_y - initial_iris_y

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

                elif (left_iris_diff > THRESHOLD_VERTICAL_D or right_iris_diff > THRESHOLD_VERTICAL_D) and time.time() - previous_time > 0.5:
                    previous_time = time.time()
                    counter = 0
                    print("Down")
                    if current_row < len(key_buttons) - 1 and current_col < len(key_buttons[current_row + 1]):
                        root.after(0, move_highlight, current_row + 1, current_col)
                        print("Down")
                        
                elif (left_iris_diff < THRESHOLD_VERTICAL_U or right_iris_diff < THRESHOLD_VERTICAL_U) and time.time() - previous_time > 1:
                    previous_time = time.time()
                    counter = 0
                    print("Up")
                    if current_row > 0:
                        root.after(0, move_highlight, current_row - 1, current_col)
                        print("Up")

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
                        

                # Visualization
                # cv2.line(frame, (0, int(initial_iris_y)), (frame.shape[1], int(initial_iris_y)), (0, 255, 255), 1)
                # cv2.circle(frame, (int(face_landmarks.landmark[473].x * frame.shape[1]), int(face_landmarks.landmark[473].y * frame.shape[0])), 3, (0, 255, 255), -1)
                # cv2.circle(frame, (int(face_landmarks.landmark[468].x * frame.shape[1]), int(face_landmarks.landmark[468].y * frame.shape[0])), 3, (0, 255, 255), -1)          
                # cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                # cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
                # cv2.putText(frame, f"Vertical Y: {int(left_iris_y-initial_iris_y)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(frame, f"Horizontal Ratio: {ratio1}  {ratio2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                # cv2.putText(frame, f"Blink Differernce: {blink_diff}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imshow("Eye Tracker", frame)
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
