import cv2
import numpy as np
from helpers import getPoint

# MediaPipe FaceMesh landmark groups
leftIrisIndexes = [468, 469, 470, 471, 472]
rightIrisIndexes = [473, 474, 475, 476, 477]

upperLipIndexes = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lowerLipIndexes = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
outerLipIndexes = upperLipIndexes + lowerLipIndexes

leftEyebrowIndexes = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
rightEyebrowIndexes = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]

# Small cheek/forehead areas. They avoid eyes, lips and strong shadows as much as possible.
skinPatchIndexes = [
    [50, 101, 118, 117, 123],      # right cheek from user's perspective after mirror flip
    [280, 330, 347, 346, 352],     # left cheek
    [10, 67, 109, 338, 297],       # forehead
    [168, 197, 5, 4, 195],         # middle face / nose bridge
]

# Created arrays of points
def getPoints(frame, faceLandmarks, indexes):
    points = []
    imageHeight, imageWidth, _ = frame.shape

    for index in indexes:
        if index >= len(faceLandmarks.landmark):
            continue

        points.append(getPoint(faceLandmarks, index, imageWidth, imageHeight))

    return np.array(points, dtype=np.int32)


def createPolygonMask(frame, points):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if points is None or len(points) < 3:
        return mask

    cv2.fillPoly(mask, [points], 255)
    return mask


def getMedianColor(frame, mask):
    pixels = frame[mask > 0]

    if len(pixels) == 0:
        return None

    medianColor = np.median(pixels, axis=0)
    return medianColor.astype(np.uint8)


def getRegionMedianColor(frame, faceLandmarks, indexes):
    points = getPoints(frame, faceLandmarks, indexes)
    mask = createPolygonMask(frame, points)
    return getMedianColor(frame, mask)


def getLuminance(bgrColor):
    if bgrColor is None:
        return None

    blue, green, red = bgrColor.astype(float)
    return 0.114 * blue + 0.587 * green + 0.299 * red


def classifyIrisColor(bgrColor):
    if bgrColor is None:
        return "Unknown"

    hsvColor = cv2.cvtColor(np.uint8([[bgrColor]]), cv2.COLOR_BGR2HSV)[0][0]
    hue = int(hsvColor[0])
    saturation = int(hsvColor[1])
    value = int(hsvColor[2])

    if value < 45:
        return "Deep dark brown"

    if value < 80 and saturation < 80:
        return "Dark brown"

    if saturation < 35:
        return "Gray"

    if hue < 18:
        return "Brown"

    if 18 <= hue < 35:
        return "Hazel"

    if 35 <= hue < 85:
        return "Green"

    if 85 <= hue < 130:
        return "Blue"

    return "Unknown"


def analyzeIrisColor(frame, faceLandmarks):
    if len(faceLandmarks.landmark) < 478:
        return {
            "label": "Unknown",
            "color": None,
            "reason": "Iris landmarks are missing. Enable refine_landmarks=True in FaceMesh."
        }

    leftColor = getRegionMedianColor(frame, faceLandmarks, leftIrisIndexes)
    rightColor = getRegionMedianColor(frame, faceLandmarks, rightIrisIndexes)

    colors = [color for color in [leftColor, rightColor] if color is not None]

    if not colors:
        return {"label": "Unknown", "color": None, "reason": "Could not read iris pixels."}

    irisColor = np.median(np.array(colors), axis=0).astype(np.uint8)
    label = classifyIrisColor(irisColor)

    return {
        "label": label,
        "color": irisColor,
        "reason": f"Median iris BGR color: {irisColor.tolist()}"
    }


def classifyLipColor(bgrColor):
    if bgrColor is None:
        return "Unknown"

    hsvColor = cv2.cvtColor(np.uint8([[bgrColor]]), cv2.COLOR_BGR2HSV)[0][0]
    labColor = cv2.cvtColor(np.uint8([[bgrColor]]), cv2.COLOR_BGR2LAB)[0][0]

    hue = int(hsvColor[0])
    saturation = int(hsvColor[1])
    value = int(hsvColor[2])
    aValue = int(labColor[1]) - 128

    if value < 70:
        return "Dark"

    if saturation < 35 and value > 150:
        return "Pale"

    if aValue > 18 and (hue < 10 or hue > 160):
        return "Red / rosy"

    if aValue > 10:
        return "Pink"

    if 8 <= hue <= 25:
        return "Peach / brownish"

    return "Neutral"


def analyzeLipColor(frame, faceLandmarks):
    lipColor = getRegionMedianColor(frame, faceLandmarks, outerLipIndexes)
    label = classifyLipColor(lipColor)

    return {
        "label": label,
        "color": lipColor,
        "reason": None if lipColor is None else f"Median lip BGR color: {lipColor.tolist()}"
    }


def getSkinMedianColor(frame, faceLandmarks):
    colors = []

    for patchIndexes in skinPatchIndexes:
        color = getRegionMedianColor(frame, faceLandmarks, patchIndexes)

        if color is not None:
            colors.append(color)

    if not colors:
        return None

    return np.median(np.array(colors), axis=0).astype(np.uint8)


def classifySkinUndertone(bgrColor):
    if bgrColor is None:
        return "Unknown"

    labColor = cv2.cvtColor(np.uint8([[bgrColor]]), cv2.COLOR_BGR2LAB)[0][0]
    aValue = int(labColor[1]) - 128
    bValue = int(labColor[2]) - 128
    difference = bValue - aValue

    if difference > 9:
        return "Warm"

    if difference < -4:
        return "Cool"

    return "Neutral"


def analyzeSkinUndertone(frame, faceLandmarks):
    skinColor = getSkinMedianColor(frame, faceLandmarks)
    label = classifySkinUndertone(skinColor)

    if skinColor is None:
        return {"label": label, "color": None, "reason": "Could not read skin pixels."}

    labColor = cv2.cvtColor(np.uint8([[skinColor]]), cv2.COLOR_BGR2LAB)[0][0]
    aValue = int(labColor[1]) - 128
    bValue = int(labColor[2]) - 128

    return {
        "label": label,
        "color": skinColor,
        "reason": f"Skin LAB a*: {aValue}, b*: {bValue}"
    }


def analyzeEyebrows(frame, faceLandmarks, skinColor):
    eyebrowMask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for eyebrowIndexes in [leftEyebrowIndexes, rightEyebrowIndexes]:
        eyebrowPoints = getPoints(frame, faceLandmarks, eyebrowIndexes)

        if len(eyebrowPoints) >= 3:
            cv2.fillPoly(eyebrowMask, [eyebrowPoints], 255)

    eyebrowPixels = frame[eyebrowMask > 0]

    if len(eyebrowPixels) == 0:
        return {
            "density": "Unknown",
            "darkness": "Unknown",
            "color": None,
            "reason": "Could not read eyebrow pixels."
        }

    eyebrowColor = np.median(eyebrowPixels, axis=0).astype(np.uint8)
    eyebrowGray = cv2.cvtColor(eyebrowPixels.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).reshape(-1)

    skinLuminance = getLuminance(skinColor)
    eyebrowLuminance = getLuminance(eyebrowColor)

    if skinLuminance is None or eyebrowLuminance is None:
        contrast = 0
    else:
        contrast = skinLuminance - eyebrowLuminance

    darkPixelThreshold = max(35, eyebrowLuminance + 8)
    darkPixelRatio = float(np.mean(eyebrowGray < darkPixelThreshold))

    if darkPixelRatio < 0.25:
        density = "Sparse"
    elif darkPixelRatio < 0.45:
        density = "Medium"
    else:
        density = "Dense"

    if contrast < 18:
        darkness = "Light"
    elif contrast < 45:
        darkness = "Medium"
    else:
        darkness = "Dark"

    return {
        "density": density,
        "darkness": darkness,
        "color": eyebrowColor,
        "reason": f"Dark pixel ratio: {darkPixelRatio:.2f}, skin-eyebrow luminance contrast: {contrast:.1f}"
    }


def drawFaceAnalysis(frame, analysisResult):
    lines = [
        f"Eyes: {analysisResult['iris']['label']}",
        f"Skin undertone: {analysisResult['skinUndertone']['label']}",
        f"Eyebrows: {analysisResult['eyebrows']['density']}, {analysisResult['eyebrows']['darkness']}",
        f"Lips: {analysisResult['lips']['label']}",
    ]

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, 40 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def faceSeg(frame, faceLandmarks):
    skinResult = analyzeSkinUndertone(frame, faceLandmarks)

    analysisResult = {
        "iris": analyzeIrisColor(frame, faceLandmarks),
        "skinUndertone": skinResult,
        "eyebrows": analyzeEyebrows(frame, faceLandmarks, skinResult["color"]),
        "lips": analyzeLipColor(frame, faceLandmarks),
    }

    drawFaceAnalysis(frame, analysisResult)
    return analysisResult
