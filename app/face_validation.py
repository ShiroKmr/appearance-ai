import cv2
import numpy as np
from helpers import getPoint

# MediaPipe indexes
NOSE_TIP = 1
LEFT_FACE = 234
RIGHT_FACE = 454
TOP_FACE = 10
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
LEFT_EYEBROW_CENTER = 105
RIGHT_EYEBROW_CENTER = 334

# Thresholds for validation
MIN_BRIGHTNESS = 100
MIN_BLUR_SCORE = 50
MIN_VISIBLE_FACE_RATIO = 0.5

# Helper to get a safe face crop from landmarks
def getFaceCrop(frame, faceLandmarks):
    imageHeight, imageWidth, _ = frame.shape

    xCoordinates = []
    yCoordinates = []

    for landmark in faceLandmarks.landmark:
        xCoordinates.append(int(landmark.x * imageWidth))
        yCoordinates.append(int(landmark.y * imageHeight))

    minX = max(min(xCoordinates), 0)
    maxX = min(max(xCoordinates), imageWidth)
    minY = max(min(yCoordinates), 0)
    maxY = min(max(yCoordinates), imageHeight)

    if minX >= maxX or minY >= maxY:
        return None

    return frame[minY:maxY, minX:maxX]

# Decide if the face is frontal
def isFaceFrontal(faceLandmarks, imageWidth, imageHeight):
    noseX, _ = getPoint(faceLandmarks, NOSE_TIP, imageWidth, imageHeight)
    leftX, _ = getPoint(faceLandmarks, LEFT_FACE, imageWidth, imageHeight)
    rightX, _ = getPoint(faceLandmarks, RIGHT_FACE, imageWidth, imageHeight)

    leftDistance = noseX - leftX
    rightDistance = rightX - noseX

    if rightDistance <= 0 or leftDistance <= 0:
        return False

    ratioX = leftDistance / rightDistance

    return 0.65 <= ratioX <= 1.35

# Check whether the face is sharp and not strongly covered
def isFaceClear(frame, faceLandmarks):
    errors = []
    faceCrop = getFaceCrop(frame, faceLandmarks)

    if faceCrop is None:
        errors.append("Face could not be cropped correctly.")
        return errors

    gray = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2GRAY)

    # Blur score strongly depends on image size and webcam quality.
    # We resize the face crop to make the threshold more stable.
    faceHeight, faceWidth = gray.shape
    if faceWidth > 0 and faceHeight > 0:
        resizedGray = cv2.resize(gray, (300, 300))
    else:
        errors.append("Face could not be checked for blur.")
        return errors

    blurScore = cv2.Laplacian(resizedGray, cv2.CV_64F).var()
    if blurScore < MIN_BLUR_SCORE:
        errors.append("Face is blurry. Please hold the camera steady.")

    _, darkMask = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY_INV)
    darkPixelRatio = np.count_nonzero(darkMask) / darkMask.size

    if darkPixelRatio > 1 - MIN_VISIBLE_FACE_RATIO:
        errors.append("Face is not clearly visible. Please remove objects covering your face.")

    return errors

# Function that puts it all together
def validateFace(frame, faceLandmarks):
    errors = []
    imageHeight, imageWidth, _ = frame.shape

    if not isFaceFrontal(faceLandmarks, imageWidth, imageHeight):
        errors.append("Please look straight into the camera.")

    errors.extend(isFaceClear(frame, faceLandmarks))

    return errors