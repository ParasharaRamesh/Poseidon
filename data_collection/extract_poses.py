import cv2
import pandas as pd
import mediapipe as mp


video_path = 'dataset/walking/1.mp4'
output_csv = 'results.csv'

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0
csv_data = []

POSE_LANDMARKS = frozenset([
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
])

POSE_CONNECTIONS = frozenset([
    (0, 11), (0, 12), # Nose-Shoulder Joints
    (11, 12), # Shoulder-Shoulder Joints
    (11, 13), (13, 15), # Left-Hand Joints
    (12, 14), (14, 16), # Right-Hand Joints
    (11, 23), (12, 24), # Shoulder-Hip Joints
    (23, 24), # Hip-Hip Joints
    (23, 25), (25, 27), # Left-Leg Joints
    (24, 26), (26, 28), # Right-Leg Joints
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, POSE_CONNECTIONS)

        # Add the landmark coordinates to the list and print them
        image_height, image_width = frame.shape[:2]
        landmarks = result.pose_landmarks.landmark
        print(f"Landmark coordinates for frame {frame_number}:")
        for idx, landmark in enumerate(landmarks):
            if idx in POSE_LANDMARKS:
                print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
                csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z, image_height, image_width])
        print("")

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(csv_data, columns=["frame_number", "landmark", "x", "y", "z", "image_height", "image_width"])
df.to_csv(output_csv, index=False)
