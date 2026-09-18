from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch import transforms
from res_insert import ResNet18, testTransform
from PIL import Image

path = Path("models/resNet18_season")
expectedModelName = "ResNet18"
defaultClassNames = ["spring", "summer", "autumn", "winter"]
defaultSubClassNames = ["bright", "cool", "deep", "light", "soft", "warm"]


class SeasonPredictor:
    def __init__(self, checkpointPath=path, predictionInterval=8, smoothingDecay=0.85, minimumPredictionCount=3, minimumConfidence=0.45):
        self.checkpointPath = Path(checkpointPath)
        self.predictionInterval = predictionInterval
        self.smoothingDecay = smoothingDecay
        self.minimumPredictionCount = minimumPredictionCount
        self.minimumConfidence = minimumConfidence

        self.validFrameCount = 0
        self.predictionCount = 0
        self.smoothedProbabilities = None

        # The model is loaded once and reused for every accepted camera frame.
        self.device = torch.device("mps")
        self.model = ResNet18()
        stateDict = torch.load(self.checkpointPath, map_location=self.device)
        self.model.load_state_dict(stateDict)
    
        self.model = self.model.to(self.device)
        self.inputTransform = testTransform

        self.classNames = list(defaultClassNames)
        self.subClassNames = list(defaultSubClassNames)

        self.loadCheckpoint()
        self.model.eval()

        self.personSegmenter = self.createPersonSegmenter()

    def createPersonSegmenter(self):
        try:
            return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except RuntimeError:
            print("Warning: hair segmentation is unavailable; using the FaceMesh mask only.")
            return None

    def calculateCropBounds(self, frame, faceLandmarks):
        imageHeight, imageWidth = frame.shape[:2]
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

        minimumX = float(xCoordinates.min())
        maximumX = float(xCoordinates.max())
        minimumY = float(yCoordinates.min())
        maximumY = float(yCoordinates.max())
        faceWidth = maximumX - minimumX
        faceHeight = maximumY - minimumY

        if faceWidth <= 1 or faceHeight <= 1:
            return None

        cropMinimumX = max(int(minimumX - 0.42 * faceWidth), 0)
        cropMaximumX = min(int(maximumX + 0.42 * faceWidth), imageWidth)
        cropMinimumY = max(int(minimumY - 0.72 * faceHeight), 0)
        cropMaximumY = min(int(maximumY + 0.18 * faceHeight), imageHeight)

        if (cropMinimumX >= cropMaximumX or cropMinimumY >= cropMaximumY):
            return None

        return cropMinimumX, cropMinimumY, cropMaximumX, cropMaximumY

    def createFaceMask(self, frame, faceLandmarks):
        imageHeight, imageWidth = frame.shape[:2]
        facePoints = np.array(
            [
                [
                    int(landmark.x * imageWidth),
                    int(landmark.y * imageHeight),
                ]
                for landmark in faceLandmarks.landmark[:468]
            ],
            dtype=np.int32,
        )
        facePoints[:, 0] = np.clip(facePoints[:, 0], 0, imageWidth - 1)
        facePoints[:, 1] = np.clip(facePoints[:, 1], 0, imageHeight - 1)

        faceMask = np.zeros((imageHeight, imageWidth), dtype=np.uint8)
        faceHull = cv2.convexHull(facePoints)
        cv2.fillConvexPoly(faceMask, faceHull, 255)

        return cv2.dilate(faceMask, np.ones((7, 7), dtype=np.uint8),iterations=1)

    def makeSquareImage(self, image):
        imageHeight, imageWidth = image.shape[:2]
        squareSize = max(imageHeight, imageWidth)
        squareImage = np.zeros((squareSize, squareSize, 3),dtype=image.dtype)

        xOffset = (squareSize - imageWidth) // 2
        yOffset = (squareSize - imageHeight) // 2
        squareImage[yOffset:yOffset + imageHeight, xOffset:xOffset + imageWidth] = image

        return squareImage

    def createMaskedHeadImage(self,frame,imageRgb,faceLandmarks):
        cropBounds = self.calculateCropBounds(frame, faceLandmarks)

        if cropBounds is None:
            return None

        cropMinimumX, cropMinimumY, cropMaximumX, cropMaximumY = cropBounds
        imageHeight, imageWidth = frame.shape[:2]

        faceMask = self.createFaceMask(frame, faceLandmarks)
        headRegionMask = np.zeros(
            (imageHeight, imageWidth),
            dtype=np.uint8,
        )
        cv2.rectangle(
            headRegionMask,
            (cropMinimumX, cropMinimumY),
            (cropMaximumX, cropMaximumY),
            255,
            thickness=-1,
        )

        # The runtime mask combines FaceMesh geometry with person segmentation so that hair remains available to the classifier.
        segmentationResult = (
            self.personSegmenter.process(imageRgb)
            if self.personSegmenter is not None
            else None
        )

        if (
            segmentationResult is not None
            and segmentationResult.segmentation_mask is not None
        ):
            personMask = (
                segmentationResult.segmentation_mask > 0.35
            ).astype(np.uint8) * 255
            headAndHairMask = cv2.bitwise_and(
                personMask,
                headRegionMask,
            )
            combinedMask = cv2.bitwise_or(faceMask, headAndHairMask)
        else:
            combinedMask = faceMask

        combinedMask = cv2.morphologyEx(
            combinedMask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8),
        )
        combinedMask = cv2.bitwise_and(combinedMask, headRegionMask)

        maskedFrame = np.zeros_like(frame)
        maskedFrame[combinedMask > 0] = frame[combinedMask > 0]
        maskedCrop = maskedFrame[
            cropMinimumY:cropMaximumY,
            cropMinimumX:cropMaximumX,
        ]

        if maskedCrop.size == 0:
            return None

        return self.makeSquareImage(maskedCrop)

    def calculateProbabilities(self, maskedHeadImage):
        rgbImage = cv2.cvtColor(maskedHeadImage, cv2.COLOR_BGR2RGB)
        inputTensor = self.inputTransform(
            Image.fromarray(rgbImage)
        ).unsqueeze(0)
        inputTensor = inputTensor.to(self.device)
        flippedTensor = torch.flip(inputTensor, dims=[3])
        inputBatch = torch.cat([inputTensor, flippedTensor], dim=0)

        with torch.inference_mode():
            logits = self.model(inputBatch).mean(dim=0)
            probabilities = torch.softmax(logits, dim=0)

        return probabilities.cpu().numpy()

    def updateSmoothing(self, probabilities):
        if self.smoothedProbabilities is None:
            self.smoothedProbabilities = probabilities
        else:
            self.smoothedProbabilities = (
                self.smoothingDecay * self.smoothedProbabilities
                + (1.0 - self.smoothingDecay) * probabilities
            )

        self.smoothedProbabilities = (
            self.smoothedProbabilities
            / self.smoothedProbabilities.sum()
        )
        self.predictionCount += 1

    def getCurrentPrediction(self):
        if self.smoothedProbabilities is None:
            return None

        classIndex = int(self.smoothedProbabilities.argmax())
        confidence = float(self.smoothedProbabilities[classIndex])

        return {
            "season": self.classNames[classIndex],
            "confidence": confidence,
            "probabilities": {
                className: float(self.smoothedProbabilities[index])
                for index, className in enumerate(self.classNames)
            },
            "predictionCount": self.predictionCount,
            "minimumPredictionCount": self.minimumPredictionCount,
            "ready": self.predictionCount >= self.minimumPredictionCount,
            "lowConfidence": confidence < self.minimumConfidence,
        }

    def processFrame(self, frame, imageRgb, faceLandmarks):
        shouldPredict = (
            self.validFrameCount % self.predictionInterval == 0
        )
        self.validFrameCount += 1

        if shouldPredict:
            maskedHeadImage = self.createMaskedHeadImage(
                frame,
                imageRgb,
                faceLandmarks,
            )

            if maskedHeadImage is not None:
                probabilities = self.calculateProbabilities(
                    maskedHeadImage
                )
                self.updateSmoothing(probabilities)

        return self.getCurrentPrediction()

    def reset(self):
        self.validFrameCount = 0
        self.predictionCount = 0
        self.smoothedProbabilities = None

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
    confidencePercent = int(round(prediction["confidence"] * 100))

    if not prediction["ready"]:
        resultText = (
            "Analyzing... "
            f"{prediction['predictionCount']}/"
            f"{prediction['minimumPredictionCount']} "
            f"({seasonName} {confidencePercent}%)"
        )
        textColor = (255, 255, 255)
    elif prediction["lowConfidence"]:
        resultText = f"Season: {seasonName}? ({confidencePercent}%)"
        textColor = (0, 200, 255)
    else:
        resultText = f"Season: {seasonName} ({confidencePercent}%)"
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
