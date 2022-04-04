import cv2

# Since each video is a bunch of images, and we already did the image face detector
# We can just loop our imageFaceDetector in every frame of the video 

# Load some pre-trained data on face frontals from opencv github (haar cascade algorithm) to create a classifier
trainedFaceData = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Choose an webcam to capture the video, in case, 0 will be our OS default webcam
# We can also give a video file as parameter like we did with image
# Our webcam variable will be used in a loop to get each frame/image of video
webcam = cv2.VideoCapture(0)

# We need a infinite loop to loop through every frame of our camera
while True:
  # Now we will read the frame and get if it was successful getting the image or not
  successfulFrameRead, frame = webcam.read()
  
  # Now we just use the code from imageFaceDetector to detect in frame
  
  # Convert to grayscaled image to have a faster algorithm
  grayscaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Detect face coordinates from grayscaled image with my classifier
  # Detect multiscale will return a array with all faces coordinates from image pre loaded
  # For each face it will return an array with x y (top left corner) and width height from x y to reach bottom right corner
  faceCoordinates = trainedFaceData.detectMultiScale(grayscaledImage)

  # For each face, I need to catch every coordinate and draw the retangle in the image
  for (x, y, w, h) in faceCoordinates:
    # First parameter is the image, the second is the coordinates of the top left corner, then the coordinate x y from bottom right corner
    # then color and last parameter is the thickness of the rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

  # Show my image with detected faces
  cv2.imshow('Face detector', frame)
  
  # We always need to have a waitkey, otherwise our app will freeze
  # So here we have waitkey to wait for 1ms to user press any key then procceed, if we dont pass any ms, it will wait
  # forever till user press a key to continue the program
  # We can receive the key pressed by the user and use one key to close our app (the key we receive come as ascii code)
  key = cv2.waitKey(1)
  
  # The code of q/Q in ascii table is 81/113 respectively, so we check it and break our while cycle if its true
  if key == 81 or key == 113:
    break

# Make sure we stop using the webcam
webcam.release()