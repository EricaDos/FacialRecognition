import opencv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = sys.argv[2]

faceCascade = opencv2.CascadeClassifier(cascPath)

#Reading the image
image = opencv2.imread(imagePath)
gray = opencv2. cvtColor(image,opencv2.COLOUR_BRG2GRAY)

#Detecting

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = opencv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    opencv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

opencv2.imshow("Faces found", image)
opencv2.waitKey(0
