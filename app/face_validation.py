# To improve:
#   1. Face frontal improve the Y axis stability
#   2. See if the classifier can work with these images
import cv2
import numpy as np

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
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 200
MIN_BLUR_SCORE = 50
MIN_VISIBLE_FACE_RATIO = 0.9
MIN_GLASSES_EDGE_RATIO = 0.035


# Helper to transform relative coordinates to pixels
def getPoint(faceLandmarks, index, imageWidth, imageHeight):
    landmark = faceLandmarks.landmark[index]
    x = int(landmark.x * imageWidth)
    y = int(landmark.y * imageHeight)
    return x, y


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


# Helper to crop the area where glasses are usually visible
def getGlassesRegion(frame, faceLandmarks):
    imageHeight, imageWidth, _ = frame.shape

    leftOuterX, leftOuterY = getPoint(faceLandmarks, LEFT_EYE_OUTER, imageWidth, imageHeight)
    rightOuterX, rightOuterY = getPoint(faceLandmarks, RIGHT_EYE_OUTER, imageWidth, imageHeight)
    leftBrowX, leftBrowY = getPoint(faceLandmarks, LEFT_EYEBROW_CENTER, imageWidth, imageHeight)
    rightBrowX, rightBrowY = getPoint(faceLandmarks, RIGHT_EYEBROW_CENTER, imageWidth, imageHeight)

    minX = max(min(leftOuterX, rightOuterX) - 20, 0)
    maxX = min(max(leftOuterX, rightOuterX) + 20, imageWidth)
    minY = max(min(leftBrowY, rightBrowY) - 15, 0)
    maxY = min(max(leftOuterY, rightOuterY) + 35, imageHeight)

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


# Check whether the lighting is acceptable
def lighting(frame, faceLandmarks=None):
    errors = []

    if faceLandmarks is not None:
        imageToCheck = getFaceCrop(frame, faceLandmarks)

        if imageToCheck is None:
            errors.append("Face could not be cropped correctly.")
            return errors
    else:
        imageToCheck = frame

    gray = cv2.cvtColor(imageToCheck, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    if brightness < MIN_BRIGHTNESS:
        errors.append("Too dark. Please improve the lighting.")

    if brightness > MAX_BRIGHTNESS:
        errors.append("Too bright. Please reduce the lighting.")

    return errors


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


# Heuristic check for glasses in the eye region
def hasGlasses(frame, faceLandmarks):
    glassesRegion = getGlassesRegion(frame, faceLandmarks)

    if glassesRegion is None:
        return False

    gray = cv2.cvtColor(glassesRegion, cv2.COLOR_BGR2GRAY)
    blurredGray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blurredGray, 50, 150)
    edgeRatio = np.count_nonzero(edges) / edges.size

    return edgeRatio > MIN_GLASSES_EDGE_RATIO


# Function that puts it all together
def validateFace(frame, faceLandmarks):
    errors = []
    imageHeight, imageWidth, _ = frame.shape

    if not isFaceFrontal(faceLandmarks, imageWidth, imageHeight):
        errors.append("Please look straight into the camera.")

    errors.extend(lighting(frame, faceLandmarks))
    errors.extend(isFaceClear(frame, faceLandmarks))

    if hasGlasses(frame, faceLandmarks):
        errors.append("Glasses detected. Please remove them for a more accurate analysis.")

    return errors