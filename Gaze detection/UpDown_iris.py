import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Function to get eye region landmarks
def get_eye_region(landmarks, indices):
    """Extract eye region landmarks."""
    return np.array([(landmarks[idx][0], landmarks[idx][1]) for idx in indices], dtype=np.int32)

# Function to calculate iris vertical position
def get_iris_position(eye_region):
    """Calculate the vertical position of the iris."""
    top = np.mean(eye_region[1:3], axis=0)
    bottom = np.mean(eye_region[4:6], axis=0)
    return (top[1] + bottom[1]) / 2

# Left eye landmark indices (based on MediaPipe FaceMesh)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Parameters for detecting significant movement
THRESHOLD = 5  # Sensitivity for detecting vertical movement

# Start video capture
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_frame)
# for face_landmarks in results.multi_face_landmarks:
    # Extract the 2D landmarks
landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
    # Get the left eye region
left_eye = get_eye_region(landmarks, LEFT_EYE_INDICES)
INITIAL_IRIS_Y = get_iris_position(left_eye)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # cv2.line(frame, (50,INITIAL_IRIS_Y),(200,INITIAL_IRIS_Y), (255,255,0), 1)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract the 2D landmarks
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

            # Get the left eye region
            left_eye = get_eye_region(landmarks, LEFT_EYE_INDICES)
            left_iris_y = get_iris_position(left_eye)

            # Compare current iris position to previous position
            if INITIAL_IRIS_Y is not None:
                movement = left_iris_y - INITIAL_IRIS_Y

                if movement > THRESHOLD:
                    pyautogui.press('down')  # Simulate down arrow key press
                    print("Down Key Pressed")
                    pyautogui.sleep(2)
                elif movement < -THRESHOLD:
                    pyautogui.press('up')  # Simulate up arrow key press
                    print("Up Key Pressed")
                    pyautogui.sleep(2)

            # Update previous iris position
            # previous_iris_y = left_iris_y

            # Draw the eye region for visualization
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)

            # Display the iris position for debugging
            cv2.putText(frame, f"Left Eye Y: {int(left_iris_y)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Initial Eye Y: {int(INITIAL_IRIS_Y)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Iris Tracker", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
