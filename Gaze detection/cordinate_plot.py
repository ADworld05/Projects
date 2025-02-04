#plot the distance of the iris to the landmarks in real-time

import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt

# Initialize video capture and face mesh
cap = cv2.VideoCapture("samplevideo2.mp4")
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Distance calculation function
def get_distance(iris, landmark, img_width, img_height):
    x1, y1 = iris.x * img_width, iris.y * img_height
    x2, y2 = landmark.x * img_width, landmark.y * img_height
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# Initialize real-time plot
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
timestamps = []
distances = {
    'left': [],
    'right': [],
    'top': [],
    'bottom': []
}

# Define colors for landmarks and plot
COLORS = {
    'iris': (0, 255, 0),       # Green for iris
    'left': (255, 0, 0),       # Blue for left landmarks
    'right': (0, 0, 255),      # Red for right landmarks
    'top': (255, 255, 0),      # Cyan for top landmarks
    'bottom': (0, 255, 255),   # Yellow for bottom landmarks
    'plot_left': 'blue',
    'plot_right': 'red',
    'plot_top': 'cyan',
    'plot_bottom': 'yellow'
}

def draw_landmark(frame, landmark, color, radius=1):
    x = int(landmark.x * frame.shape[1])
    y = int(landmark.y * frame.shape[0])
    cv2.circle(frame, (x, y), radius, color, -1)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize frame for processing
    resized_frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
    img_height, img_width = resized_frame.shape[:2]
    
    # Process frame with face mesh
    results = face_mesh.process(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Right eye landmarks
        right_iris = landmarks[473]
        right_landmarks = {
            'left': landmarks[362],
            'right': landmarks[263],
            'top': landmarks[386],
            'bottom': landmarks[374]
        }

        # Draw right eye landmarks
        draw_landmark(resized_frame, right_iris, COLORS['iris'], radius=1)
        for key, landmark in right_landmarks.items():
            draw_landmark(resized_frame, landmark, COLORS[key], radius=1)

        # Calculate all distances for right eye
        current_time = time.time()
        dist_left = get_distance(right_iris, right_landmarks['left'], img_width, img_height)
        dist_right = get_distance(right_iris, right_landmarks['right'], img_width, img_height)
        dist_top = get_distance(right_iris, right_landmarks['top'], img_width, img_height)
        dist_bottom = get_distance(right_iris, right_landmarks['bottom'], img_width, img_height)

        # Update data
        timestamps.append(current_time)
        distances['left'].append(dist_left)
        distances['right'].append(dist_right)
        distances['top'].append(dist_top)
        distances['bottom'].append(dist_bottom)

        # Clear and redraw plot
        ax.clear()
        ax.plot(timestamps, distances['left'], color=COLORS['plot_left'], label='Left Distance')
        ax.plot(timestamps, distances['right'], color=COLORS['plot_right'], label='Right Distance')
        ax.plot(timestamps, distances['top'], color=COLORS['plot_top'], label='Top Distance')
        ax.plot(timestamps, distances['bottom'], color=COLORS['plot_bottom'], label='Bottom Distance')
        
        ax.set_title("Right Iris to Landmark Distances")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (pixels)")
        ax.legend()
        plt.pause(0.001)

    # Display video feed
    cv2.imshow('Eye Tracking', resized_frame)
    if cv2.waitKey(10) == ord('q'):
        break

# Cleanup
plt.ioff()
plt.show()
cap.release()
cv2.destroyAllWindows()