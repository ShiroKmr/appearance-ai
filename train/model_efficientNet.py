import torch
from torch import nn

from torchvision.models import (
    EfficientNet_V2_S_Weights,
    efficientnet_v2_s,
)


numberOfClasses = 4
numberOfSubClasses = 6


def getDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class MaskedColorStatistics(nn.Module):
    """Preserve global colour cues that an ImageNet backbone can underuse."""

    numberOfStatistics = 16

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "imageNetMean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imageNetStandardDeviation",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def calculateMoments(self, values, validMask, validCount):
        mean = (values * validMask).sum(dim=(2, 3)) / validCount
        centeredValues = values - mean.unsqueeze(-1).unsqueeze(-1)
        variance = (
            centeredValues.square() * validMask
        ).sum(dim=(2, 3)) / validCount

        return mean, variance.clamp_min(1e-8).sqrt()

    def forward(self, normalizedImages):
        images = (
            normalizedImages * self.imageNetStandardDeviation
            + self.imageNetMean
        ).clamp(0.0, 1.0)

        validMask = (
            images.amax(dim=1, keepdim=True) > 0.035
        ).to(images.dtype)
        validCount = validMask.sum(dim=(2, 3)).clamp_min(1.0)

        rgbMean, rgbStandardDeviation = self.calculateMoments(
            images,
            validMask,
            validCount,
        )

        channelSum = images.sum(dim=1, keepdim=True).clamp_min(1e-6)
        chromaticity = images / channelSum
        chromaticityMean, chromaticityStandardDeviation = (
            self.calculateMoments(
                chromaticity,
                validMask,
                validCount,
            )
        )

        maximumChannel = images.amax(dim=1, keepdim=True)
        minimumChannel = images.amin(dim=1, keepdim=True)
        saturation = (
            (maximumChannel - minimumChannel)
            / maximumChannel.clamp_min(1e-6)
        )

        valueMean, valueStandardDeviation = self.calculateMoments(
            maximumChannel,
            validMask,
            validCount,
        )
        saturationMean, saturationStandardDeviation = self.calculateMoments(
            saturation,
            validMask,
            validCount,
        )

        return torch.cat(
            [
                rgbMean,
                rgbStandardDeviation,
                chromaticityMean,
                chromaticityStandardDeviation,
                valueMean,
                valueStandardDeviation,
                saturationMean,
                saturationStandardDeviation,
            ],
            dim=1,
        )


class SeasonColorModel(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        backbone = efficientnet_v2_s(weights=weights)
        backboneFeatureCount = backbone.classifier[1].in_features

        self.features = backbone.features
        self.averagePool = backbone.avgpool
        self.colorStatistics = MaskedColorStatistics()

        combinedFeatureCount = (
            backboneFeatureCount
            + self.colorStatistics.numberOfStatistics
        )

        self.classifier = nn.Sequential(
            nn.Linear(combinedFeatureCount, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(p=0.35),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(p=0.20),
        )
        self.seasonHead = nn.Linear(256, numberOfClasses)
        self.subClassHead = nn.Linear(256, numberOfSubClasses)

        self.initializeClassificationLayers()

    def initializeClassificationLayers(self):
        for module in [
            self.classifier[0],
            self.classifier[4],
            self.seasonHead,
            self.subClassHead,
        ]:
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def extractFeatures(self, images):
        visualFeatures = self.features(images)
        visualFeatures = self.averagePool(visualFeatures)
        visualFeatures = torch.flatten(visualFeatures, 1)

        colorFeatures = self.colorStatistics(images)

        return self.classifier(
            torch.cat([visualFeatures, colorFeatures], dim=1)
        )

    def forward(self, images):
        sharedFeatures = self.extractFeatures(images)

        return self.seasonHead(sharedFeatures)

    def forwardWithAuxiliary(self, images):
        sharedFeatures = self.extractFeatures(images)

        return (
            self.seasonHead(sharedFeatures),
            self.subClassHead(sharedFeatures),
        )


def loadPretrainedModel(usePretrainedWeights=True):
    weights = EfficientNet_V2_S_Weights.DEFAULT
    modelWeights = weights if usePretrainedWeights else None

    model = SeasonColorModel(weights=modelWeights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    return model, weights
