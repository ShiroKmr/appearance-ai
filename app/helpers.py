# Functions that are used across multiple files
import cv2
import numpy as np

# Add a vignette to the image
# def addWhiteVignette(frame, strength):
#     height, width = frame.shape[:2]

#     heightMask = cv2.getGaussianKernel(height, height/2)
#     widthMask = cv2.getGaussianKernel(width, width/2)

#     newFrame = heightMask @ widthMask.T
#     newFrame = newFrame / newFrame.max()

#     edge = 1 - newFrame
#     edge = np.clip(edge * strength, 0, 1)

#     whiteFrame = np.full_like(frame, 255)
#     edge = edge[:, :, np.newaxis]

#     result = frame.astype(np.float32) * (1 - edge) + whiteFrame.astype(np.float32) * edge
#     return result.astype(np.uint8)

# Helper to transform relative coordinates to pixels
def getPoint(faceLandmarks, index, imageWidth, imageHeight):
    landmark = faceLandmarks.landmark[index]
    x = int(landmark.x * imageWidth)
    y = int(landmark.y * imageHeight)
    return x, y

# Created arrays of points
def getPoints(frame, faceLandmarks, indexes):
    points = []
    imageHeight, imageWidth, _ = frame.shape

    for index in indexes:
        if index >= len(faceLandmarks.landmark):
            continue

        points.append(getPoint(faceLandmarks, index, imageWidth, imageHeight))

    return np.array(points, dtype=np.int32)

# Get median color of a region
def getRegionMedianColor(frame, faceLandmarks, indexes):
    points = getPoints(frame, faceLandmarks, indexes)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if points is None or len(points) < 3:
        return mask

    cv2.fillPoly(mask, [points], 255)
    pixels = frame[mask > 0]

    if len(pixels) == 0:
        return None

    medianColor = np.median(pixels, axis=0)
    return medianColor

def getLuminance(bgrColor):
    if bgrColor is None:
        return None

    blue, green, red = bgrColor.astype(float)
    # Scale the values closer to how people percieve them
    return 0.114 * blue + 0.587 * green + 0.299 * red