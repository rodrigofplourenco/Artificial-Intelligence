import cv2

# Load some pre-trained data on face frontals from opencv github (haar cascade algorithm) to create a classifier
trainedFaceData = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Choose an image to detect a face
image = cv2.imread('images/elon.png')

# Convert to grayscaled image to have a faster algorithm
grayscaledImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect face coordinates from grayscaled image with my classifier
# Detect multiscale will return a array with all faces coordinates from image pre loaded
# For each face it will return an array with x y (top left corner) and width height from x y to reach bottom right corner
faceCoordinates = trainedFaceData.detectMultiScale(grayscaledImage)

# For each face, I need to catch every coordinate and draw the retangle in the image
for (x, y, w, h) in faceCoordinates:
  # First parameter is the image, the second is the coordinates of the top left corner, then the coordinate x y from bottom right corner
  # then color and last parameter is the thickness of the rectangle
  cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

# Show my image with detected faces
cv2.imshow('Face detector', image)

# Wait till user press a key to continue the program
cv2.waitKey()
