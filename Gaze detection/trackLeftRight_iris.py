import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize video capture
cam = cv2.VideoCapture(0)

# Initialize Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Variables to store the minimum differences
min_left_diff = float('inf')  # Start with a very large number
min_right_diff = float('inf')

while True:
    _, frame = cam.read()
    
    # Convert frame to RGB
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    # Process landmarks if detected
    if output.multi_face_landmarks:
        for face_landmarks in output.multi_face_landmarks:
            # Scale landmarks to frame size
            iris_x = int(face_landmarks.landmark[473].x * frame.shape[1])
            left_x = int(face_landmarks.landmark[362].x * frame.shape[1])
            right_x = int(face_landmarks.landmark[263].x * frame.shape[1])

            # Calculate differences
            left_diff = abs(iris_x - left_x)
            right_diff = abs(iris_x - right_x)

            ratio = round(left_diff / right_diff, 3)

            if ratio < 0.8:
                cv2.putText(frame, "Looking left", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # pyautogui.press('left')
            elif ratio > 1.6:
                cv2.putText(frame, "Looking right", (30, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # pyautogui.press('right')
            else:
                cv2.putText(frame, "Looking straight", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            

            # Draw landmarks for visualization
            cv2.circle(frame, (iris_x, int(face_landmarks.landmark[473].y * frame.shape[0])), 3, (0, 255, 0), -1)
            cv2.circle(frame, (left_x, int(face_landmarks.landmark[362].y * frame.shape[0])), 3, (255, 0, 0), -1)
            cv2.circle(frame, (right_x, int(face_landmarks.landmark[263].y * frame.shape[0])), 3, (0, 0, 255), -1)
            
            #left detection
            cv2.circle(frame, (int(face_landmarks.landmark[159].x * frame.shape[1]), int(face_landmarks.landmark[159].y * frame.shape[0])), 3, (0, 255, 255), -1)         
            cv2.circle(frame, (int(face_landmarks.landmark[145].x * frame.shape[1]), int(face_landmarks.landmark[145].y * frame.shape[0])), 3, (0, 255, 255), -1)          

            blink_ratio = int(face_landmarks.landmark[145].y * frame.shape[0]) - int(face_landmarks.landmark[159].y * frame.shape[0])
            cv2.putText(frame, f"Blink Ratio: {blink_ratio}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2) 

            if blink_ratio < 5:
                cv2.putText(frame, "Blinking", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # pyautogui.click()

            # Display calculated distances
            cv2.putText(frame, f"Left Diff: {left_diff}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Right Diff: {right_diff}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Ratio: {ratio}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display the frame
    frame_resized = cv2.resize(frame, None, fx =1.5, fy =1.5)
    cv2.imshow('Iris Tracker', frame_resized)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cam.release()
cv2.destroyAllWindows()

# Print minimum differences
print(f"Minimum Left Difference: {min_left_diff}")
print(f"Minimum Right Difference: {min_right_diff}")
