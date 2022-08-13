import cv2

# Load pre-trained data on face frontals from opencv (hear casade algorithm)
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# Detect a face in an image
img = cv2.imread('billionairs.jpg')

# convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around each face in image
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

#
cv2.imshow('Face Detector', img)
cv2.waitKey()


# Capture video from webcam
webcam = cv2.VideoCapture('guy.mp4')

# Iterate over each frame from webcam
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    if successful_frame_read == False:
        break
    # Convert frames to grayscale
    grayscaled_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    vid_coordinates = trained_face_data.detectMultiScale(grayscaled_vid)

    for (x, y, w, h) in vid_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Capture video from file
webcam = cv2.VideoCapture(0)

# Iterate over each frame in webcam
while True:

    # Read the current frame
    successful_frame_read, frame = webcam.read()

    # Convert frames to grayscale
    grayscaled_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    vid_coordinates = trained_face_data.detectMultiScale(grayscaled_vid)

    for (x, y, w, h) in vid_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# release the VideoCapture object
webcam.release()

print("Code Completed")
