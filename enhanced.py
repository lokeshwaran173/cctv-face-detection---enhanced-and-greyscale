import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the MP4 file
video_capture = cv2.VideoCapture('mp4 fomat/073816-00087-M-0012.mp4')  # Replace 'your_video.mp4' with the path to your MP4 file

frame_skip = 5  # Skip every 5 frames
frame_count = 0

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

    # Display the frames
    cv2.imshow('Original', original_frame)
    cv2.imshow('Grayscale', gray_frame)
    cv2.imshow('Enhanced', enhanced_frame)
    cv2.imshow('Original with Face Detection', original_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display windows
video_capture.release()
cv2.destroyAllWindows()
