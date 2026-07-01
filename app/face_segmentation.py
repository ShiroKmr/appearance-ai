import cv2
import numpy as np
from helpers import getPoints, getRegionMedianColor, getLuminance

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
        return {
            "label": "Unknown",
            "color": None,
            "reason": "Could not read iris pixels."
        }

    irisColor = np.median(np.array(colors), axis=0).astype(np.uint8)

    hsvColor = cv2.cvtColor(np.uint8([[irisColor]]), cv2.COLOR_BGR2HSV)[0][0]
    hue = int(hsvColor[0])
    saturation = int(hsvColor[1])
    value = int(hsvColor[2])

    label = "Unknown"
    print(hue, " ", saturation, " ", value, "\n")
    if value < 45:
        label = "Deep dark brown"
    elif value < 80 and saturation < 80:
        label = "Dark brown"
    elif saturation < 35:
        label = "Gray"
    elif hue < 10:
        label = "Brown"
    elif 10 <= hue < 27:
        label = "Hazel"
    elif 27 <= hue < 78:
        label = "Green"
    elif 78 <= hue < 130:
        label = "Blue"

    return {
        "label": label,
        "color": irisColor,
        "reason": f"Median iris BGR color: {irisColor.tolist()}"
    }


def classifyLipColor(frame, faceLandmarks):
    if faceLandmarks is None or frame is None:
        return {
            "label": "Unknown",
            "color": None,
            "reason": "Frame or face landmarks are missing."
        }

    meanColor = getRegionMedianColor(frame, faceLandmarks, outerLipIndexes)

    if meanColor is None:
        return {
            "label": "Unknown",
            "color": None,
            "reason": "Could not read lip pixels."
        }

    # HSV = hue, saturation, value
    hsvColor = cv2.cvtColor(np.uint8([[meanColor]]), cv2.COLOR_BGR2HSV)[0][0]
    labColor = cv2.cvtColor(np.uint8([[meanColor]]), cv2.COLOR_BGR2LAB)[0][0]

    hue = int(hsvColor[0])
    saturation = int(hsvColor[1])
    value = int(hsvColor[2])
    aValue = int(labColor[1]) - 128

    lipColor = "Neutral"

    # The numbers are heuristic thresholds and should be calibrated on real camera frames.
    if value < 70:
        lipColor = "Dark"
    elif saturation < 35 and value > 150:
        lipColor = "Pale"
    elif aValue > 18 and (hue < 10 or hue > 160):
        lipColor = "Red / rosy"
    elif aValue > 10:
        lipColor = "Pink"
    elif 8 <= hue <= 25:
        lipColor = "Peach / brownish"

    return {
        "label": lipColor,
        "color": meanColor,
        "reason": f"Median lip BGR color: {meanColor.tolist()}"
    }


def analyzeSkinUndertone(frame, faceLandmarks):
    skinColors = []

    for skinPatchIndex in skinPatchIndexes:
        skinPatchColor = getRegionMedianColor(frame, faceLandmarks, skinPatchIndex)

        if skinPatchColor is not None:
            skinColors.append(skinPatchColor.astype(np.float32))

    skinColor = np.median(np.array(skinColors), axis=0).astype(np.uint8)

    # LAB = lightness, a: green/red, b: blue/yellow
    labColor = cv2.cvtColor(np.uint8([[skinColor]]), cv2.COLOR_BGR2LAB)[0][0]
    aValue = int(labColor[1]) - 128
    bValue = int(labColor[2]) - 128
    difference = bValue - aValue

    if difference > 9:
        label = "Warm"
    elif difference < -4:
        label = "Cool"
    else:
        label = "Neutral"

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

    # Compare the eyebrows with the skin luminance to determine what is dark and what is bright.
    if skinLuminance is not None:
        darkPixelThreshold = max(30, skinLuminance - 25)
    else:
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
    skin = analyzeSkinUndertone(frame, faceLandmarks)

    analysisResult = {
        "iris": analyzeIrisColor(frame, faceLandmarks),
        "skinUndertone": skin,
        "eyebrows": analyzeEyebrows(frame, faceLandmarks, skin["color"]),
        "lips": classifyLipColor(frame, faceLandmarks),
    }

    drawFaceAnalysis(frame, analysisResult)
    return analysisResult
