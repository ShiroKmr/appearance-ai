# Functions that are used across multiple files

# Helper to transform relative coordinates to pixels
def getPoint(faceLandmarks, index, imageWidth, imageHeight):
    landmark = faceLandmarks.landmark[index]
    x = int(landmark.x * imageWidth)
    y = int(landmark.y * imageHeight)
    return x, y

