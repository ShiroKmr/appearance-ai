# Loading the model
import torch
from torch import nn

from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)

numberOfClasses = 4

def getDevice():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def loadPretrainedModel():
    weights = EfficientNet_B0_Weights.DEFAULT

    model = efficientnet_b0(
        weights=weights,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False
    
    inputFeatures = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=inputFeatures,
        out_features=numberOfClasses,
    )

    return model, weights
