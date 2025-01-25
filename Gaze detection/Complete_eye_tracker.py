import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
)

# Function to get eye region landmarks
def get_eye_region(landmarks, indices):
    """Extract eye region landmarks."""
    return np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in indices], dtype=np.int32)

# Function to calculate iris vertical position
def get_iris_vertical_position(eye_region):
    """Calculate the vertical position of the iris."""
    top = np.mean(eye_region[1:3], axis=0)
    bottom = np.mean(eye_region[4:6], axis=0)
    return (top[1] + bottom[1]) / 2

# Function to calculate iris horizontal position
def get_iris_horizontal_position(iris_x, left_x, right_x):
    """Calculate the horizontal position of the iris."""
    left_diff = abs(iris_x - left_x)
    right_diff = abs(iris_x - right_x)
    return left_diff, right_diff

# Landmark indices for eyes and iris (based on MediaPipe FaceMesh)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
BLINK_TOP = 159
BLINK_BOTTOM = 145
IRIS_LEFT = 362
IRIS_RIGHT = 263

# Parameters
THRESHOLD_VERTICAL_U = 4  # Sensitivity for detecting vertical movement
THRESHOLD_VERTICAL_D = 6 # Sensitivity for detecting vertical movement
THRESHOLD_HORIZONTAL_L = 0.8  # Ratio sensitivity for left detection
THRESHOLD_HORIZONTAL_R = 1.4  # Ratio sensitivity for right detection


# Start video capture
cap = cv2.VideoCapture(0)

# Get initial iris position for up-down tracking
ret, frame = cap.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)

if results.multi_face_landmarks:
    landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
    left_eye = get_eye_region(landmarks, LEFT_EYE_INDICES)
    INITIAL_IRIS_Y = get_iris_vertical_position(left_eye)

    # Sensitivity for blink detection
    top_y = int(results.multi_face_landmarks[0].landmark[BLINK_TOP].y * frame.shape[0])
    bottom_y = int(results.multi_face_landmarks[0].landmark[BLINK_BOTTOM].y * frame.shape[0])
    BLINK_THRESHOLD = (bottom_y - top_y)/2    
else:
    INITIAL_IRIS_Y = None
    BLINK_THRESHOLD = None

previous_time = time.time()

# Clock function for periodic actions
def clock(previous_time, interval):
    current_time = time.time()
    return current_time - previous_time > interval

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

            # Left eye for up-down and left-right tracking
            left_eye = get_eye_region(landmarks, LEFT_EYE_INDICES)
            left_iris_y = get_iris_vertical_position(left_eye)

            # Iris horizontal position
            iris_x = int(face_landmarks.landmark[473].x * frame.shape[1])
            left_x = int(face_landmarks.landmark[IRIS_LEFT].x * frame.shape[1])
            right_x = int(face_landmarks.landmark[IRIS_RIGHT].x * frame.shape[1])
            left_diff, right_diff = get_iris_horizontal_position(iris_x, left_x, right_x)
            ratio = round(left_diff / right_diff, 3)

            # Blink detection
            top_y = int(face_landmarks.landmark[BLINK_TOP].y * frame.shape[0])
            bottom_y = int(face_landmarks.landmark[BLINK_BOTTOM].y * frame.shape[0])
            blink_diff = bottom_y - top_y

            # Detect blinking
            if blink_diff < BLINK_THRESHOLD and clock(previous_time, interval=2):
                previous_time = time.time()
                # pyautogui.click()
                print("Blink Detected")

            # Detect vertical movement (up-down)
            if INITIAL_IRIS_Y is not None:
                vertical_movement = left_iris_y - INITIAL_IRIS_Y

                if vertical_movement > THRESHOLD_VERTICAL_D  and clock(previous_time, interval=2):
                    previous_time = time.time()
                    # pyautogui.press("down")
                    print("Down Key Pressed")
                elif vertical_movement < -THRESHOLD_VERTICAL_U  and clock(previous_time, interval=2):
                    previous_time = time.time()
                    # pyautogui.press("up")
                    print("Up Key Pressed")

            # Detect horizontal movement (left-right)
            if ratio < THRESHOLD_HORIZONTAL_L and clock(previous_time, interval=2):
                previous_time = time.time()
                # pyautogui.press("left")
                print("Left Key Pressed")
            elif ratio > THRESHOLD_HORIZONTAL_R and clock(previous_time, interval=2):  # Adjust ratio threshold for right
                previous_time = time.time()
                # pyautogui.press("right")
                print("Right Key Pressed")

            # Visualization
            cv2.circle(frame, (int(face_landmarks.landmark[159].x * frame.shape[1]), int(face_landmarks.landmark[159].y * frame.shape[0])), 3, (0, 255, 255), -1)         
            cv2.circle(frame, (int(face_landmarks.landmark[145].x * frame.shape[1]), int(face_landmarks.landmark[145].y * frame.shape[0])), 3, (0, 255, 255), -1)          
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.putText(frame, f"Vertical Y: {int(left_iris_y-INITIAL_IRIS_Y)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Horizontal Ratio: {ratio}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Blink Differernce: {blink_diff}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("4-Way Eye Movement & Blink Tracker", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
