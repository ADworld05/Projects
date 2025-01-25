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
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
BLINK_TOP = 159
BLINK_BOTTOM = 145
IRIS_LEFT = 362
IRIS_RIGHT = 263
THRESHOLD_VERTICAL_U = 4
THRESHOLD_VERTICAL_D = 6
THRESHOLD_HORIZONTAL_L = 0.8
THRESHOLD_HORIZONTAL_R = 1.4

# Keyboard setup
root = tk.Tk()
root.title("Iris-Controlled Virtual Keyboard")
root.geometry("720x440")

input_field = tk.Entry(root, font=("Arial", 18), width=40)
input_field.grid(row=0, column=0, columnspan=10, pady=10)

keys = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
    ["z", "x", "c", "v", "b", "n", "m"],
]

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
            command=lambda char=key: input_field.insert(tk.END, char),
        )
        btn.grid(row=row + 1, column=col, padx=5, pady=5)
        button_row.append(btn)
    key_buttons.append(button_row)

space_button = tk.Button(
    root,
    text="Space",
    font=("Arial", 14),
    width=15,
    height=2,
    bg="lightgray",
    command=lambda: input_field.insert(tk.END, " "),
)
space_button.grid(row=5, column=0, columnspan=5, pady=5)

backspace_button = tk.Button(
    root,
    text="Backspace",
    font=("Arial", 14),
    width=15,
    height=2,
    bg="lightgray",
    command=lambda: input_field.delete(len(input_field.get()) - 1),
)
backspace_button.grid(row=5, column=5, columnspan=5, pady=5)

key_buttons.append([space_button, backspace_button])

current_row, current_col = 0, 0
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
    global current_row, current_col
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
        initial_iris_y = np.mean([landmarks[159][1], landmarks[145][1]])
        top_y = landmarks[BLINK_TOP][1]
        bottom_y = landmarks[BLINK_BOTTOM][1]
        blink_threshold = (bottom_y - top_y) / 2
    else:
        initial_iris_y = None
        blink_threshold = None

    previous_time = time.time()

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

                left_iris_y = np.mean([landmarks[159][1], landmarks[145][1]])
                iris_x = landmarks[473][0]
                left_x = landmarks[IRIS_LEFT][0]
                right_x = landmarks[IRIS_RIGHT][0]
                left_diff = abs(iris_x - left_x)
                right_diff = abs(iris_x - right_x)
                ratio = left_diff / right_diff

                left_eye = get_eye_region(landmarks, LEFT_EYE_INDICES)
                top_y = landmarks[BLINK_TOP][1]
                bottom_y = landmarks[BLINK_BOTTOM][1]
                blink_diff = bottom_y - top_y

                if blink_diff < blink_threshold and time.time() - previous_time > 2:
                    previous_time = time.time()
                    root.after(0, blink_action)

                vertical_movement = left_iris_y - initial_iris_y
                if vertical_movement > THRESHOLD_VERTICAL_D and time.time() - previous_time > 2:
                    previous_time = time.time()
                    if current_row < len(key_buttons) - 1:
                        root.after(0, move_highlight, current_row + 1, current_col)
                elif vertical_movement < -THRESHOLD_VERTICAL_U and time.time() - previous_time > 2:
                    previous_time = time.time()
                    if current_row > 0:
                        root.after(0, move_highlight, current_row - 1, current_col)

                if ratio < THRESHOLD_HORIZONTAL_L and time.time() - previous_time > 2:
                    previous_time = time.time()
                    if current_col > 0:
                        root.after(0, move_highlight, current_row, current_col - 1)
                elif ratio > THRESHOLD_HORIZONTAL_R and time.time() - previous_time > 2:
                    previous_time = time.time()
                    if current_col < len(key_buttons[current_row]) - 1:
                        root.after(0, move_highlight, current_row, current_col + 1)

                # Visualization
                cv2.circle(frame, (int(face_landmarks.landmark[159].x * frame.shape[1]), int(face_landmarks.landmark[159].y * frame.shape[0])), 3, (0, 255, 255), -1)         
                cv2.circle(frame, (int(face_landmarks.landmark[145].x * frame.shape[1]), int(face_landmarks.landmark[145].y * frame.shape[0])), 3, (0, 255, 255), -1)          
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
                cv2.putText(frame, f"Vertical Y: {int(left_iris_y-initial_iris_y)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Horizontal Ratio: {ratio}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Blink Differernce: {blink_diff}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Eye Tracker", frame)
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