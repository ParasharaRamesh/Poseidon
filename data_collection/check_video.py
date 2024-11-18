import cv2
import mediapipe as mp


video_path = 'sample.mp4'

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_number = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920 * 2, 1080))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_image = frame.copy()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Add the landmark coordinates to the list and print them
        image_height, image_width = frame.shape[:2]
        landmarks = result.pose_landmarks.landmark
        print(f"Landmark coordinates for frame {frame_number}:")
        for idx, landmark in enumerate(landmarks):
            print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        print("")

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1
    annotated_image = frame.copy()
    combined_image = cv2.hconcat([original_image, annotated_image])
    out.write(combined_image)

cap.release()
out.release()
cv2.destroyAllWindows()
