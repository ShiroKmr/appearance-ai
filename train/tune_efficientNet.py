# Fine-tune the model
import torch
from torch import nn
from torch.optim import Adam

from training_helpers import model, device, classifierCheckpointPath, trainEpochs

numberOfEpochs = 20

backboneLearningRate = 0.00001
classifierLearningRate = 0.0001

def loadCheckpoint(model, checkpointPath, device):
    if not checkpointPath.is_file():
        raise FileNotFoundError(
            "Classifier checkpoint was not found:\n"
            f"{checkpointPath}"
        )

    checkpoint = torch.load(checkpointPath, map_location=device)

    if (isinstance(checkpoint, dict) and "modelState" in checkpoint):
        modelState = checkpoint["modelState"]
    else:
        modelState = checkpoint

    model.load_state_dict(modelState)

    return model

def unfreezeLastBlocks(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.features[3].parameters():
        parameter.requires_grad = True

    for parameter in model.features[4].parameters():
        parameter.requires_grad = True

    for parameter in model.features[5].parameters():
        parameter.requires_grad = True

    for parameter in model.features[6].parameters():
        parameter.requires_grad = True

    for parameter in model.features[7].parameters():
        parameter.requires_grad = True

    for parameter in model.features[8].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model

def printTrainableParameters(model):
    print("Trainable parameters:")

    trainableParameterCount = 0

    for parameterName, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainableParameterCount += parameter.numel()

    print(
        "\nNumber of trainable scalar parameters: "
        f"{trainableParameterCount}"
    )


model = loadCheckpoint(
    model=model,
    checkpointPath=classifierCheckpointPath,
    device=device,
)

model = unfreezeLastBlocks(model)

printTrainableParameters(model)

lossFunction = nn.CrossEntropyLoss()
optimizer = Adam(
    [
        {
            "params": model.features[3].parameters(),
            "lr": backboneLearningRate,
        },

        {
            "params": model.features[4].parameters(),
            "lr": backboneLearningRate,
        },
        {
            "params": model.features[5].parameters(),
            "lr": backboneLearningRate,
        },
        {
            "params": model.features[6].parameters(),
            "lr": backboneLearningRate,
        },

        {
            "params": model.features[7].parameters(),
            "lr": backboneLearningRate,
        },
        {
            "params": model.features[8].parameters(),
            "lr": backboneLearningRate,
        },
        {
            "params": model.classifier.parameters(),
            "lr": classifierLearningRate,
        },
    ]
)

tuneIndicator = True

bestAcc = trainEpochs(numberOfEpochs, lossFunction, optimizer, model, tuneIndicator)
print("Best validation accuracy: ", bestAcc)
