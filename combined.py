import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the MP4 file
video_capture = cv2.VideoCapture('mp4 fomat/073816-00087-M-0012.mp4')  # Replace 'your_video.mp4' with the path to your MP4 file

frame_skip = 5  # Skip every 5 frames
frame_count = 0

# Calculate the size for the frames to be displayed
combined_height, combined_width = 720, 1280  # Adjust these values as needed
frame_height, frame_width = combined_height // 2, combined_width // 2

while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    frame_count += 1

    if not ret:
        break  # Break the loop when the video ends

    if frame_count % frame_skip != 0:
        continue

    # Create a copy of the original frame
    original_frame = frame.copy()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance the frame (adjust contrast and brightness)
    alpha = 1.1  # Slight contrast adjustment
    beta = 10    # Slight brightness adjustment
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Convert the enhanced frame to grayscale for face detection
    gray_enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the original frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    # Resize all frames to the same dimensions
    original_frame = cv2.resize(original_frame, (frame_width, frame_height))
    gray_frame = cv2.resize(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR), (frame_width, frame_height))
    enhanced_frame = cv2.resize(enhanced_frame, (frame_width, frame_height))
    gray_enhanced_frame = cv2.resize(cv2.cvtColor(gray_enhanced_frame, cv2.COLOR_GRAY2BGR), (frame_width, frame_height))

    # Combine the frames into one image
    combined_frame = np.vstack(
        (np.hstack((original_frame, gray_frame)),
         np.hstack((enhanced_frame, gray_enhanced_frame))
        )
    )

    # Display the combined frame
    cv2.imshow('Combined Frames', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
video_capture.release()
cv2.destroyAllWindows()
