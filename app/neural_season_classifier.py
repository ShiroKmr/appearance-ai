from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from app.resnet import ResNet18, testTransform
from PIL import Image

path = Path("models/resNet18_season.pth")
expectedModelName = "ResNet18"
defaultClassNames = ["spring", "summer", "autumn", "winter"]
defaultSubClassNames = ["bright", "cool", "deep", "light", "soft", "warm"]


class SeasonPredictor:
    def __init__(self, checkpointPath=path, predictionInterval=8, smoothingDecay=0.7, minimumPredictionCount=3, minimumConfidence=0.45):
        self.checkpointPath = Path(checkpointPath)
        self.predictionInterval = predictionInterval
        self.smoothingDecay = smoothingDecay
        self.minimumPredictionCount = minimumPredictionCount
        self.minimumConfidence = minimumConfidence

        self.validFrameCount = 0
        self.predictionCount = 0
        self.smoothedClassProbabilities = None
        self.smoothedSubClassProbabilities = None

        # The model is loaded once and reused for every accepted camera frame.
        self.device = torch.device("mps")
        self.model = ResNet18()
        stateDict = torch.load(self.checkpointPath, map_location=self.device)
        self.model.load_state_dict(stateDict)
    
        self.model = self.model.to(self.device)
        self.inputTransform = testTransform

        self.classNames = list(defaultClassNames)
        self.subClassNames = list(defaultSubClassNames)

        self.model.eval()

        self.personSegmenter = self.createPersonSegmenter()

    # Create an automatic person segmenter, including hair, shoulders, etc
    def createPersonSegmenter(self):
        try: # Choose the landscape model, import selfie segmentation from MediaPipe to separate the face from the background
            return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except RuntimeError:
            print("Warning: hair segmentation is unavailable; using the FaceMesh mask only.")
            return None

    # Сrop the face (as a rectangle) and return the colored version of it
    def fillFaceRectangle(self, frame, faceLandmarks):
        imageHeight, imageWidth = frame.shape[:2]
        # Scale the coordinates bavk to pixels and not relative numbers, do so for every landmark
        xCoordinates = np.array(
            [
                landmark.x * imageWidth
                for landmark in faceLandmarks.landmark
            ]
        )
        yCoordinates = np.array(
            [
                landmark.y * imageHeight
                for landmark in faceLandmarks.landmark
            ]
        )

        # Choose the min and max to find the boarders of the face and calculate its width and height
        minimumX = float(xCoordinates.min())
        maximumX = float(xCoordinates.max())
        minimumY = float(yCoordinates.min())
        maximumY = float(yCoordinates.max())
        faceWidth = maximumX - minimumX
        faceHeight = maximumY - minimumY

        if faceWidth <= 1 or faceHeight <= 1:
            return None

        # Move the boarders to the sides, so some hair is visible
        cropMinimumX = max(int(minimumX - 0.42 * faceWidth), 0)
        cropMaximumX = min(int(maximumX + 0.42 * faceWidth), imageWidth)
        cropMinimumY = max(int(minimumY - 0.72 * faceHeight), 0)
        cropMaximumY = min(int(maximumY + 0.18 * faceHeight), imageHeight)

        if (cropMinimumX >= cropMaximumX or cropMinimumY >= cropMaximumY):
            return None

        headRegionMask = np.zeros((imageHeight, imageWidth),dtype=np.uint8)
        return  cv2.rectangle(
                    headRegionMask,
                    (cropMinimumX, cropMinimumY),
                    (cropMaximumX, cropMaximumY),
                    255,
                    thickness=-1,
                ), cropMinimumX, cropMaximumX, cropMinimumY, cropMaximumY

    # Pad the image
    def makeSquareImage(self, image):
        imageHeight, imageWidth = image.shape[:2]
        squareSize = max(imageHeight, imageWidth)
        squareImage = np.zeros((squareSize, squareSize, 3),dtype=image.dtype)

        xOffset = (squareSize - imageWidth) // 2
        yOffset = (squareSize - imageHeight) // 2
        squareImage[yOffset:yOffset + imageHeight, xOffset:xOffset + imageWidth] = image

        return squareImage

    def createMaskedHeadImage(self,frame,imageRgb,faceLandmarks):
        headMask, cropMinimumX, cropMaximumX, cropMinimumY, cropMaximumY = self.fillFaceRectangle(frame, faceLandmarks)

        # Probability map where there is a person and where is not
        segmentationResult = self.personSegmenter.process(imageRgb)

        # Choose only the pixels, for which the confidence is bigger than 35%. Fill them with white
        personMask = (segmentationResult.segmentation_mask > 0.35).astype(np.uint8) * 255

        # Delete unesesary things like shoulders and the background from the personMask
        headAndHairMask = cv2.bitwise_and(personMask,headMask)

        # Fill the empty spaces
        headAndHairMask = cv2.morphologyEx(
            headAndHairMask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        headAndHairMask = cv2.bitwise_and(headAndHairMask, headMask)

        maskedFrame = np.zeros_like(frame)
        maskedFrame[headAndHairMask > 0] = frame[headAndHairMask > 0]
        maskedCrop = maskedFrame[cropMinimumY:cropMaximumY,cropMinimumX:cropMaximumX]

        if maskedCrop.size == 0:
            return None

        return self.makeSquareImage(maskedCrop)

    def calculateProbabilities(self, maskedHeadImage):
        # Transform to RGB
        rgbImage = cv2.cvtColor(maskedHeadImage, cv2.COLOR_BGR2RGB)

        # Transform the rgbImage to PIL.Image, transform the same as inputs for the ResNet, unsqueeze adds the batch dimension
        inputTensor = self.inputTransform(Image.fromarray(rgbImage)).unsqueeze(0)
        inputTensor = inputTensor.to(self.device)

        # Flip the face horizontally and combine them into one batch
        flippedTensor = torch.flip(inputTensor, dims=[3])
        inputBatch = torch.cat([inputTensor, flippedTensor], dim=0)

        # Test-Time Augmentation: take the mean from the predictions to ensure that they're more stable
        with torch.inference_mode():
            classLogits, subClassLogits = self.model(inputBatch)

            classLogits = classLogits.mean(dim=0)
            subClassLogits = subClassLogits.mean(dim=0)

            classProbabilities = torch.softmax(classLogits, dim=0)
            subClassProbabilities = torch.softmax(subClassLogits, dim=0)

        return classProbabilities.cpu().numpy(), subClassProbabilities.cpu().numpy()

    def updateSmoothing(self, classProbabilities, subClassProbabilities):
        # If this is the first prediction, just take it
        # If it's not the first, then add them to the previous predictions with less weight
        if self.smoothedClassProbabilities is None and self.smoothedSubClassProbabilities is None:
            self.smoothedClassProbabilities = classProbabilities
            self.smoothedSubClassProbabilities = subClassProbabilities
        else:
            self.smoothedClassProbabilities = self.smoothingDecay * self.smoothedClassProbabilities + (1.0 - self.smoothingDecay) * classProbabilities
            self.smoothedSubClassProbabilities = self.smoothingDecay * self.smoothedSubClassProbabilities + (1.0 - self.smoothingDecay) * subClassProbabilities

        # Normalize the probabilities back
        self.smoothedClassProbabilities = self.smoothedClassProbabilities / self.smoothedClassProbabilities.sum()
        self.smoothedSubClassProbabilities = self.smoothedSubClassProbabilities / self.smoothedSubClassProbabilities.sum()    
        self.predictionCount += 1

    def getCurrentPrediction(self):
        if (self.smoothedClassProbabilities is None or self.smoothedSubClassProbabilities is None):
            return None

        classIndex = int(self.smoothedClassProbabilities.argmax())
        subClassIndex = int(self.smoothedSubClassProbabilities.argmax())

        classConfidence = float(self.smoothedClassProbabilities[classIndex])
        subClassConfidence = float(self.smoothedSubClassProbabilities[subClassIndex])

        return {
            "season": self.classNames[classIndex],
            "subClass": self.subClassNames[subClassIndex],

            "classConfidence": classConfidence,
            "subClassConfidence": subClassConfidence,

            "classProbabilities": {
                className: float(self.smoothedClassProbabilities[index])
                for index, className in enumerate(self.classNames)
            },

            "subClassProbabilities": {
                subClassName: float(self.smoothedSubClassProbabilities[index])
                for index, subClassName in enumerate(self.subClassNames)
            },
            "predictionCount": self.predictionCount,
            "minimumPredictionCount": self.minimumPredictionCount,
            "ready": self.predictionCount >= self.minimumPredictionCount,

            "lowClassConfidence": classConfidence < self.minimumConfidence,

            "lowSubClassConfidence": subClassConfidence < self.minimumConfidence
        }

    def processFrame(self, frame, imageRgb, faceLandmarks):
        shouldPredict = (self.validFrameCount % self.predictionInterval == 0)
        self.validFrameCount += 1

        if shouldPredict:
            maskedHeadImage = self.createMaskedHeadImage(frame,imageRgb,faceLandmarks)

            classProbabilities, subClassProbabilities = self.calculateProbabilities(maskedHeadImage)
            self.updateSmoothing(classProbabilities, subClassProbabilities)

        return self.getCurrentPrediction()

    def reset(self):
        self.validFrameCount = 0
        self.predictionCount = 0
        self.smoothedClassProbabilities = None
        self.smoothedSubClassProbabilities = None

    def close(self):
        if self.personSegmenter is not None:
            self.personSegmenter.close()


def drawSeasonPrediction(frame, prediction):
    if prediction is None:
        cv2.putText(
            frame,
            "Analyzing color season...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return

    seasonName = prediction["season"].title()
    subSeasonName = prediction["subClass"].title()
    confidencePercent = int(round(prediction["classConfidence"] * 100))
    subConfidencePercent = int(round(prediction["subClassConfidence"] * 100))

    if not prediction["ready"]:
        resultText = (
            "Analyzing... "
            f"{prediction['predictionCount']}/"
            f"{prediction['minimumPredictionCount']} "
            f"({seasonName} {confidencePercent}%)"
            f"({subSeasonName} {subConfidencePercent}%)"
        )
        textColor = (255, 255, 255)
    elif prediction["lowClassConfidence"]:
        resultText = (f"Season: {seasonName}? ({confidencePercent}%)"
                      f"SubSeason: {subSeasonName}? ({subConfidencePercent}%)")
        textColor = (0, 200, 255)
    else:
        resultText = (f"Season: {seasonName} ({confidencePercent}%)"
                      f"SubSeason: {subSeasonName} ({subConfidencePercent}%)")
        textColor = (100, 255, 100)

    cv2.rectangle(frame, (10, 10), (570, 60), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        resultText,
        (20, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        textColor,
        2,
    )
