#right iris [474:478]

import cv2
import mediapipe as mp
import time

cam=cv2.VideoCapture(0)
face_mesh=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

while True:
    _,frame=cam.read()
    frame = cv2.flip(frame, 1)
    
    rgb_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    
    

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Create an empty list to store the coordinates
        landmark_coords = []

        # Loop through the specified landmarks (from 474 to 477)
        for i in range(474, 478):
            landmark = landmarks[i]
            x = int(landmark.x * frame.shape[1])  # Convert to pixel x
            y = int(landmark.y * frame.shape[0])  # Convert to pixel y

            # Append the coordinates as a tuple to the list
            landmark_coords.append((x, y))

            # Draw the circle on the image
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        print(f"({landmark_coords[0][0]}, {landmark_coords[0][1]}), "
              f"({landmark_coords[1][0]}, {landmark_coords[1][1]}), "
              f"({landmark_coords[2][0]}, {landmark_coords[2][1]}), "
              f"({landmark_coords[3][0]}, {landmark_coords[3][1]})")
          
            
   

    cv2.imshow('Eye Iris Detection',frame) 
    if cv2.waitKey(1) == ord('q'):
        break


