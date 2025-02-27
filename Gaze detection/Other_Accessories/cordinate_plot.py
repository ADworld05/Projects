#plot of landamarks and iris diffrence in real time

import cv2
import mediapipe as mp
import time
import matplotlib
# Set backend before importing pyplot
#matplotlib.use('TkAgg')  # Use Tkinter backend for better interaction
import matplotlib.pyplot as plt

# Initialize video capture and face mesh
cap = cv2.VideoCapture("samplevideo.mp4")
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

def get_distance(iris, landmark, img_width, img_height):
    x1, y1 = iris.x * img_width, iris.y * img_height
    x2, y2 = landmark.x * img_width, landmark.y * img_height
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# Initialize plots with explicit figure management
plt.ion()
fig_right, ax_right = plt.subplots(figsize=(12, 4))
fig_left, ax_left = plt.subplots(figsize=(12, 4))
fig_right.canvas.manager.set_window_title('Right Eye Distances')
fig_left.canvas.manager.set_window_title('Left Eye Distances')

# Data storage
timestamps = []
distances_right = {'left': [], 'right': [], 'top': [], 'bottom': []}
distances_left = {'left': [], 'right': [], 'top': [], 'bottom': []}

# Color configuration (unchanged)
COLORS = {
    'iris': (0, 255, 0), 'right_left': (255, 0, 0), 'right_right': (0, 0, 255),
    'right_top': (255, 255, 0), 'right_bottom': (0, 255, 255),
    'left_left': (255, 165, 0), 'left_right': (128, 0, 128),
    'left_top': (255, 192, 203), 'left_bottom': (0, 128, 0),
    'plot_right_left': 'blue', 'plot_right_right': 'red',
    'plot_right_top': 'cyan', 'plot_right_bottom': 'yellow',
    'plot_left_left': 'orange', 'plot_left_right': 'purple',
    'plot_left_top': 'pink', 'plot_left_bottom': 'green'
}

def draw_landmark(frame, landmark, color, radius=2):
    x = int(landmark.x * frame.shape[1])
    y = int(landmark.y * frame.shape[0])
    cv2.circle(frame, (x, y), radius, color, -1)

def draw_eye_data(frame, distances, x_start, color_prefix, eye_label):
    y_start = 30
    line_height = 25
    cv2.putText(frame, eye_label, (x_start, y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, direction in enumerate(['left', 'right', 'top', 'bottom']):
        y = y_start + (i + 1) * line_height
        current_distance = distances[direction][-1] if distances[direction] else 0.0
        color = COLORS[f"{color_prefix}{direction}"]
        cv2.putText(frame, f"{direction.capitalize()}: {current_distance:.2f}", 
                    (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def update_plot(ax, distances, title, colors):
    ax.clear()
    ax.plot(timestamps, distances['left'], color=colors[0], label='Left')
    ax.plot(timestamps, distances['right'], color=colors[1], label='Right')
    ax.plot(timestamps, distances['top'], color=colors[2], label='Top')
    ax.plot(timestamps, distances['bottom'], color=colors[3], label='Bottom')
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (pixels)")
    ax.legend()

while True:
    success, frame = cap.read()
    if not success:
        break

    resized_frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
    img_height, img_width = resized_frame.shape[:2]
    
    results = face_mesh.process(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        current_time = time.time()
        timestamps.append(current_time)

        # Right Eye Processing (unchanged)
        right_iris = landmarks[473]
        right_landmarks = {
            'left': landmarks[362], 'right': landmarks[263],
            'top': landmarks[386], 'bottom': landmarks[374]
        }
        for direction in ['left', 'right', 'top', 'bottom']:
            distances_right[direction].append(
                get_distance(right_iris, right_landmarks[direction], img_width, img_height)
            )
            draw_landmark(resized_frame, right_landmarks[direction], COLORS[f'right_{direction}'])

        # Left Eye Processing (unchanged)
        left_iris = landmarks[468]
        left_landmarks = {
            'left': landmarks[33], 'right': landmarks[133],
            'top': landmarks[159], 'bottom': landmarks[145]
        }
        for direction in ['left', 'right', 'top', 'bottom']:
            distances_left[direction].append(
                get_distance(left_iris, left_landmarks[direction], img_width, img_height)
            )
            draw_landmark(resized_frame, left_landmarks[direction], COLORS[f'left_{direction}'])

        # Draw iris landmarks
        draw_landmark(resized_frame, right_iris, COLORS['iris'], 2)
        draw_landmark(resized_frame, left_iris, COLORS['iris'], 2)

        # Display text data
        draw_eye_data(resized_frame, distances_right, 10, 'right_', "Right Eye:")
        draw_eye_data(resized_frame, distances_left, img_width - 150, 'left_', "Left Eye:")

        # Update plots with non-blocking method
        update_plot(ax_right, distances_right, "Right Iris Distances", 
                   [COLORS['plot_right_left'], COLORS['plot_right_right'], 
                    COLORS['plot_right_top'], COLORS['plot_right_bottom']])
        update_plot(ax_left, distances_left, "Left Iris Distances",
                   [COLORS['plot_left_left'], COLORS['plot_left_right'],
                    COLORS['plot_left_top'], COLORS['plot_left_bottom']])

        # Non-blocking plot updates
        for fig in [fig_right, fig_left]:
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.001)

    # Display video with faster refresh
    cv2.imshow('Eye Tracking', resized_frame)
    key = cv2.waitKey(1)  # Critical change: Reduced from 100ms to 1ms
    if key == ord('q'):
        break

# Cleanup
plt.close('all')
cap.release()
cv2.destroyAllWindows()