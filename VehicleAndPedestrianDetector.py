import cv2 

# Load some pre-trained data on vehicles (haar cascade algorithm) to create a classifier
trainedBusData = cv2.CascadeClassifier('data/haarcascade_bus.xml')
trainedCarData = cv2.CascadeClassifier('data/haarcascade_car.xml')
trainedPedestrianData = cv2.CascadeClassifier('data/haarcascade_pedestrian.xml')

# Get a video to proccess each frame and detect vehicles or pedestrians
video = cv2.VideoCapture('videos/front_dash_cam.mp4')

# Loop through each frame of the video
while True:
  # Now we will read each frame and get if it was successful getting the image or not
  successfulFrameRead, frame = video.read()
  
  # Convert to grayscaled image to have a faster algorithm
  grayscaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
  # Detect vehicle and pedestrian coordinates from grayscaled image with my classifiers
  # Detect multiscale will return a array with all vehicles/pedestrians coordinates from image pre loaded
  # For each vehicle/pedestrian it will return an array with x y (top left corner) and width height from x y to reach bottom right corner
  pedestrianCoordinates = trainedPedestrianData.detectMultiScale(grayscaledImage)
  
  carCoordinates = trainedCarData.detectMultiScale(grayscaledImage)
  busCoordinates = trainedBusData.detectMultiScale(grayscaledImage)

  # For each pedestrian, I need to catch every coordinate and draw the retangle in the image
  for (x, y, w, h) in pedestrianCoordinates:
    # First parameter is the image, the second is the coordinates of the top left corner, then the coordinate x y from bottom right corner
    # then color and last parameter is the thickness of the rectangle
    if h in range(65, 81):
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

  # For each car, I need to catch every coordinate and draw the retangle in the image
  for (x, y, w, h) in carCoordinates:
    # First parameter is the image, the second is the coordinates of the top left corner, then the coordinate x y from bottom right corner
    # then color and last parameter is the thickness of the rectangle
    if w > 65:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
  # For each bus, I need to catch every coordinate and draw the retangle in the image
  for (x, y, w, h) in busCoordinates:
    # First parameter is the image, the second is the coordinates of the top left corner, then the coordinate x y from bottom right corner
    # then color and last parameter is the thickness of the rectangle
    if w > 65:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

  # Show my image with detected vehicles and pedestrians
  cv2.imshow('Vehicles/pedestrians detector', frame)
  
  # We always need to have a waitkey, otherwise our app will freeze
  # So here we have waitkey to wait for 1ms to user press any key then procceed, if we dont pass any ms, it will wait
  # forever till user press a key to continue the program
  # We can receive the key pressed by the user and use one key to close our app (the key we receive come as ascii code)
  key = cv2.waitKey(1)
  
  # The code of q/Q in ascii table is 81/113 respectively, so we check it and break our while cycle if its true
  if key == 81 or key == 113:
    break

# Make sure we stop using the video
video.release()