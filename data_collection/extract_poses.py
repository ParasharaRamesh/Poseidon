import os

import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

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

ACTIONS = ["arm_stretch", "leg_stretch", "lunges", "side_stretch", "walking"]

# Initialse lists to store poses and actions labels
train_2d_poses = []
train_3d_poses = []
train_actions = []

test_2d_poses = []
test_3d_poses = []
test_actions = []

dataset_base_dir = "dataset"
for action in ACTIONS:
    dataset_action_dir = os.path.join(dataset_base_dir, action)

    for video_idx in range(1, 17):
        video_path = os.path.join(dataset_action_dir, f"{video_idx}.mp4")
        print(video_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        pose_2ds = []
        pose_3ds = []
        pose_actions = []

        frame_number = 0
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
                pose_landmarks = result.pose_landmarks.landmark
                pose_world_landmarks = result.pose_world_landmarks.landmark
                print(f"Landmark coordinates for frame {frame_number}:")
                pose_2d = []
                pose_3d = []
                for idx, (pose_landmark, pose_world_landmark) in enumerate(zip(pose_landmarks, pose_world_landmarks)):
                    if idx in POSE_LANDMARKS:
                        print(f"{mp_pose.PoseLandmark(idx).name}")
                        assert pose_landmark.visibility > 0.4
                        print(pose_landmark.x , pose_landmark.y)
                        pose_2d.append((pose_landmark.x , pose_landmark.y))
                        print(pose_world_landmark.x , pose_world_landmark.y, pose_world_landmark.z)
                        pose_3d.append((pose_world_landmark.x , pose_world_landmark.y, pose_world_landmark.z))
                pose_2ds.append(pose_2d)
                pose_3ds.append(pose_3d)
                pose_actions.append(action)
                print("")

            # Display the frame
            cv2.imshow('MediaPipe Pose', frame)

            # Exit if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1
        
        train_2d_poses.extend(pose_2ds[:-35])
        train_3d_poses.extend(pose_3ds[:-35])
        train_actions.extend(pose_actions[:-35])

        test_2d_poses.extend(pose_2ds[-35:])
        test_3d_poses.extend(pose_3ds[-35:])
        test_actions.extend(pose_actions[-35:])

cap.release()
cv2.destroyAllWindows()

train_actions = np.array(train_actions)
train_2d_poses = np.array(train_2d_poses)
train_3d_poses = np.array(train_3d_poses)

test_actions = np.array(test_actions)
test_2d_poses = np.array(test_2d_poses)
test_3d_poses = np.array(test_3d_poses)

print(train_actions.shape, train_2d_poses.shape, train_3d_poses.shape)
print(test_actions.shape, test_2d_poses.shape, test_3d_poses.shape)

np.save("train_actions.npy", train_actions)
np.save("train_2d_poses.npy", train_2d_poses)
np.save("train_3d_poses.npy", train_3d_poses)

np.save("test_actions.npy", test_actions)
np.save("test_2d_poses.npy", test_2d_poses)
np.save("test_3d_poses.npy", test_3d_poses)
