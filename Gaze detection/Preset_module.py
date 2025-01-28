

previous_time = time.time()
max_l = 0
max_y = 0

while time.time()-previous_time < 5:
    ret, frame = cap.read()
    if not ret:
            break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
        initial_iris_y = np.mean([landmarks[159][1], landmarks[145][1]])
        top_y = landmarks[BLINK_TOP][1]
        bottom_y = landmarks[BLINK_BOTTOM][1]
        if()
        blink_threshold = (bottom_y - top_y)*0.5
    else:
        initial_iris_y = None
        blink_threshold = None
        print("Face ")
