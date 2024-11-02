import cv2

cap_identifier = 'mac'
image_fourcc = 'MJPG'
image_width = 1920
image_height = 1080
image_fps = 30
CAMERA_API_PREFERENCES = {"mac": cv2.CAP_AVFOUNDATION}

# Define the video capture object for the default webcam (ID=0)
cap = cv2.VideoCapture(0, apiPreference=CAMERA_API_PREFERENCES.get(cap_identifier, cv2.CAP_ANY))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*image_fourcc))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)
cap.set(cv2.CAP_PROP_FPS, image_fps)

# Set video codec and output file
fourcc = cv2.VideoWriter_fourcc(*image_fourcc)
out = cv2.VideoWriter('output.mp4', fourcc, image_fps, (image_width, image_height))

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    print("Recording video. Press 'q' to stop.")

# Capture and record video
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Write frame to file
        out.write(frame)
        print(frame.shape)
        
        # Display the frame
        cv2.imshow('Recording', frame)

        # Exit recording when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()