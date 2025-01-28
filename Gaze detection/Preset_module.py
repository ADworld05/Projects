

previous_time = time.time()
max_ratio = 0
min_ratio = 10

while time.time()-previous_time < 5:
    if results.multi_face_landmarks:
        landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.multi_face_landmarks[0].landmark]
        iris_x = landmarks[473][0]
        left_x = landmarks[IRIS_LEFT][0]
        right_x = landmarks[IRIS_RIGHT][0]
        left_diff = abs(iris_x - left_x)
        right_diff = abs(iris_x - right_x)
        ratio = left_diff / right_diff
        if(ratio>max_ratio):
            max_ratio = ratio
        if(ratio<min_ratio):
            min_ratio = ratio
        left_threshold = min_ratio*2/3 + max_ratio*1/3
        right_threshold = min_ratio*1/3 + max_ratio*2/3
