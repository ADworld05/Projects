# to tract the distance between iris and the landmarks 

# to display coordinates on the eye region in graph form
# measure relative distance from landmarks
import cv2
import mediapipe as mp
import time
import numpy as np
import tkinter as tk

def insert_char(char):
    """Inserts the clicked character into the input field."""
    input_field.insert(tk.END, char)

def backspace():
    """Deletes the last character in the input field."""
    input_field.delete(len(input_field.get()) - 1, tk.END)

# Create the main application window
root = tk.Tk()
root.title("Custom Virtual Keyboard")
root.geometry("720x440")

# Input field
input_field = tk.Entry(root, font=("Arial", 18), width=40)
input_field.grid(row=0, column=0, columnspan=10, pady=10)

# Keyboard layout
keys = [
    ["a", "b", "c", "d", "e"],  # Row 1
    ["f", "g", "h", "i", "j"],  # Row 2
    ["k", "l", "m", "n", "o"],  # Row 3
    ["p", "q", "r", "s", "t"],  # Row 4
    ["u", "v", "w", "x", "y", "z"],  # Row 5
]

# Create buttons and store references
key_buttons = []
for row, key_row in enumerate(keys, start=1):
    button_row = []
    for col, key in enumerate(key_row):
        btn = tk.Button(
            root, text=key, font=("Arial", 14), width=4, height=2, bg="lightgray",
            command=lambda char=key: insert_char(char))
        btn.grid(row=row, column=col, padx=5, pady=5)
        button_row.append(btn)
    key_buttons.append(button_row)

# Add Backspace button at the end of Row 3
backspace_button = tk.Button(
    root, text="⌫", font=("Arial", 14), width=4, height=2, bg="lightgray", command=backspace
)
backspace_button.grid(row=3, column=5, padx=5, pady=5)
key_buttons[2].append(backspace_button)  # Add to Row 3 for navigation

# Add Space button at the end of Row 4
space_button = tk.Button(
    root, text="__", font=("Arial", 14), width=4, height=2, bg="lightgray", command=lambda: insert_char(" ")
)
space_button.grid(row=4, column=5, columnspan=2, padx=5, pady=5)
key_buttons[3].append(space_button)  # Add to Row 4 for navigation

# Initial highlighted key
current_row, current_col = 0, 0
key_buttons[current_row][current_col].config(bg="yellow")

def move_highlight(new_row, new_col):
    """Move the highlight to the new position."""
    global current_row, current_col
    key_buttons[current_row][current_col].config(bg="lightgray")  # Reset previous
    current_row, current_col = new_row, new_col
    key_buttons[current_row][current_col].config(bg="yellow")  # Highlight new

def handle_keypress(event):
    """Handle arrow key presses to move the highlighted key."""
    global current_row, current_col
    if event.keysym == "Left" and current_col > 0:
        move_highlight(current_row, current_col - 1)
    elif event.keysym == "Right" and current_col < len(key_buttons[current_row]) - 1:
        move_highlight(current_row, current_col + 1)
    elif event.keysym == "Up" and current_row > 0:
        move_highlight(current_row - 1, min(current_col, len(key_buttons[current_row - 1]) - 1))
    elif event.keysym == "Down" and current_row < len(key_buttons) - 1:
        move_highlight(current_row + 1, min(current_col, len(key_buttons[current_row + 1]) - 1))

def select_highlighted_key(event):
    """Simulate a button press for the highlighted key."""
    key_buttons[current_row][current_col].invoke()  # Simulates a button click

# Bind Enter key to trigger the highlighted button
root.bind("<Return>", select_highlighted_key)

# Bind arrow keys to movement
root.bind("<Left>", handle_keypress)
root.bind("<Right>", handle_keypress)
root.bind("<Up>", handle_keypress)
root.bind("<Down>", handle_keypress)

# Run the application
root.mainloop()


cap = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def get_distance(iris, eye_landmark):
    x1 = iris.x * frame.shape[1]
    y1 = iris.y * frame.shape[0]
    x2 = eye_landmark.x * frame.shape[1]
    y2 = eye_landmark.y * frame.shape[0]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

distances_right = {'left': [], 'right': [], 'top': [], 'bottom': []}
distances_left = {'left': [], 'right': [], 'top': [], 'bottom': []}


while True:
    _, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #img_height, img_width = frame.shape[:2]
    output_frame = face_mesh.process(rgb_frame)
    landmark_points = output_frame.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        right_iris = landmarks[473]
        right_landmarks = {
            'left': landmarks[362], 'right': landmarks[263],
            'top': landmarks[386], 'bottom': landmarks[374]
        }
        for direction in right_landmarks:
            distances_right[direction].append(
                get_distance(right_iris, right_landmarks[direction])
            )

        # Left Eye Processing
        left_iris = landmarks[468]
        left_landmarks = {
            'left': landmarks[33], 'right': landmarks[133],
            'top': landmarks[159], 'bottom': landmarks[145]
        }
        for direction in left_landmarks:
            distances_left[direction].append(
                get_distance(left_iris, left_landmarks[direction])
            )  

        print(distances_right)
        print(distances_left)
        time.sleep(1) 
             
   
    cv2.imshow('Eye Iris Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

