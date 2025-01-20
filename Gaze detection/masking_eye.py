#isolating the eye using mask
#hence comprehend the iris movement

import cv2
import mediapipe as mp
import numpy as np


cam=cv2.VideoCapture(0)
#initialize the face mesh
face_mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=4) 
#initialize the drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

while True:
    _,frame=cam.read()
    
    #changing frame color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    
    #getting eye coordinates
    if output.multi_face_landmarks:
        for face_landmarks in output.multi_face_landmarks:
            right_eye = np.array([(face_landmarks.landmark[362].x * frame.shape[1], face_landmarks.landmark[362].y * frame.shape[0]),
                                 (face_landmarks.landmark[382].x * frame.shape[1], face_landmarks.landmark[382].y * frame.shape[0]),
                                 (face_landmarks.landmark[381].x * frame.shape[1], face_landmarks.landmark[381].y * frame.shape[0]),
                                 (face_landmarks.landmark[380].x * frame.shape[1], face_landmarks.landmark[380].y * frame.shape[0]),
                                 (face_landmarks.landmark[374].x * frame.shape[1], face_landmarks.landmark[374].y * frame.shape[0]),
                                 (face_landmarks.landmark[373].x * frame.shape[1], face_landmarks.landmark[373].y * frame.shape[0]),
                                 (face_landmarks.landmark[390].x * frame.shape[1], face_landmarks.landmark[390].y * frame.shape[0]),
                                 (face_landmarks.landmark[249].x * frame.shape[1], face_landmarks.landmark[249].y * frame.shape[0]),
                                 (face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]),
                                 (face_landmarks.landmark[466].x * frame.shape[1], face_landmarks.landmark[466].y * frame.shape[0]),
                                 (face_landmarks.landmark[388].x * frame.shape[1], face_landmarks.landmark[388].y * frame.shape[0]),
                                 (face_landmarks.landmark[387].x * frame.shape[1], face_landmarks.landmark[387].y * frame.shape[0]),
                                 (face_landmarks.landmark[386].x * frame.shape[1], face_landmarks.landmark[386].y * frame.shape[0]),
                                 (face_landmarks.landmark[385].x * frame.shape[1], face_landmarks.landmark[385].y * frame.shape[0]),
                                 (face_landmarks.landmark[384].x * frame.shape[1], face_landmarks.landmark[384].y * frame.shape[0]),
                                 (face_landmarks.landmark[398].x * frame.shape[1], face_landmarks.landmark[398].y * frame.shape[0])], np.int32)

            left_eye = np.array([(face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]),
                                  (face_landmarks.landmark[7].x * frame.shape[1], face_landmarks.landmark[7].y * frame.shape[0]),
                                  (face_landmarks.landmark[163].x * frame.shape[1], face_landmarks.landmark[163].y * frame.shape[0]),
                                  (face_landmarks.landmark[144].x * frame.shape[1], face_landmarks.landmark[144].y * frame.shape[0]),
                                  (face_landmarks.landmark[145].x * frame.shape[1], face_landmarks.landmark[145].y * frame.shape[0]),
                                  (face_landmarks.landmark[153].x * frame.shape[1], face_landmarks.landmark[153].y * frame.shape[0]),
                                  (face_landmarks.landmark[154].x * frame.shape[1], face_landmarks.landmark[154].y * frame.shape[0]),
                                  (face_landmarks.landmark[155].x * frame.shape[1], face_landmarks.landmark[155].y * frame.shape[0]),
                                  (face_landmarks.landmark[133].x * frame.shape[1], face_landmarks.landmark[133].y * frame.shape[0]),
                                  (face_landmarks.landmark[173].x * frame.shape[1], face_landmarks.landmark[173].y * frame.shape[0]),
                                  (face_landmarks.landmark[157].x * frame.shape[1], face_landmarks.landmark[157].y * frame.shape[0]),
                                  (face_landmarks.landmark[158].x * frame.shape[1], face_landmarks.landmark[158].y * frame.shape[0]),
                                  (face_landmarks.landmark[159].x * frame.shape[1], face_landmarks.landmark[159].y * frame.shape[0]),
                                  (face_landmarks.landmark[160].x * frame.shape[1], face_landmarks.landmark[160].y * frame.shape[0]),
                                  (face_landmarks.landmark[161].x * frame.shape[1], face_landmarks.landmark[161].y * frame.shape[0]),
                                  (face_landmarks.landmark[246].x * frame.shape[1], face_landmarks.landmark[246].y * frame.shape[0])], np.int32)

            #cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
            #cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

            #apply mask to the eye
            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(frame, [right_eye], True, (0,255,255), 2)
            cv2.fillPoly(mask, [right_eye], 255)
            right_eye_mask = cv2.bitwise_and(rgb_frame, rgb_frame, mask=mask)

            #get the min and max coordinates of the left eye
            min_x = np.min(right_eye[:, 0])
            min_y = np.min(right_eye[:, 1])
            max_x = np.max(right_eye[:, 0])
            max_y = np.max(right_eye[:, 1])

            #get the eye frame
            eye_frame=frame[min_y:max_y,min_x:max_x]
            eye_frame=cv2.resize(eye_frame, None, fx=5,fy=5)
            cv2.imshow('Eye Frame',eye_frame)  

            #convert the eye frame to gray
            gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
            cv2.imshow('Threshold Eye', threshold_eye) 

            #apply mask and isolate the eye
            cv2.imshow('right_eye_mask', right_eye_mask)          
    

    cv2.imshow('Eye Iris Detection',frame) 
    if cv2.waitKey(1) == ord('q'):
        break


