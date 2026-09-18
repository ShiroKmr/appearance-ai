# Functions that are used across multiple files
import cv2
import numpy as np

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