import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Indices for iris landmarks (MediaPipe Face Mesh model)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_TOP_BOTTOM = [386, 374]  # Top and bottom of the left eye
RIGHT_EYE_TOP_BOTTOM = [159, 145]  # Top and bottom of the right eye
LEFT_EYEBROW = [70]  # Approximate top of left eyebrow
RIGHT_EYEBROW = [300]  # Approximate top of right eyebrow
NOSE_TIP = [1]  # Nose tip landmark

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
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
            
            print(int(left_iris_diff*100),int(right_iris_diff*100))
            # Determine movement
            movement = "Center"
            if left_iris_diff < 0.15 or right_iris_diff < 0.15:
                movement = "Looking Up"
            elif left_iris_diff > 0.26 or right_iris_diff > 0.26:
                movement = "Looking Down"
            
            cv2.putText(frame, movement, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw iris centers
            cv2.circle(frame, tuple(left_iris_center), 2, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_iris_center), 2, (0, 255, 0), -1)
    
    cv2.imshow('Iris Tracking', frame)
    
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
