# Train the model
from torch import nn
from torch.optim import Adam
from model_efficientNet import getDevice, loadPretrainedModel

# Work with a dataset
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SeasonDataset

# Validation
import torch
from sklearn.metrics import f1_score

# Device: mps
# Loss function: CrossEntropyLoss()
# Optimizer: Adam (
# Parameter Group 0
#     amsgrad: False
#     betas: (0.9, 0.999)
#     capturable: False
#     decoupled_weight_decay: False
#     differentiable: False
#     eps: 1e-08
#     foreach: None
#     fused: None
#     lr: 0.001
#     maximize: False
#     weight_decay: 0
# )
# Optimizer parameter count: 5124

# Define the hyperparameters
learningRate = 0.001

device = getDevice()

model, weights = loadPretrainedModel()
model = model.to(device)

lossFunction = nn.CrossEntropyLoss()

optimizer = Adam(
    model.classifier.parameters(),
    lr=learningRate,
)

# Find the path to the dataset
projectRoot = Path(__file__).resolve().parents[1]

releaseRoot = (
    projectRoot
    / "assets"
    / "deep_armocromia"
    / "release"
)

annotationsPath = (
    releaseRoot
    / "annotations_with_validation.csv"
)

batchSize = 32

transform = weights.transforms()

trainDataset = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="train",
    transform=transform,
)

trainLoader = DataLoader(
    dataset=trainDataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers=0,
)

# Set one epoch
def calculateCorrectPredictions(predictions, labels):
    predictedClasses = predictions.argmax(dim=1)

    correctPredictions = (
        predictedClasses == labels
    ).sum().item()

    return correctPredictions


def trainOneEpoch(model, dataLoader, lossFunction, optimizer, device):
    model.train()

    totalLoss = 0.0
    totalCorrect = 0
    totalSamples = 0

    for images, labels in dataLoader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = lossFunction(
            predictions,
            labels,
        )

        loss.backward()
        optimizer.step()

        currentBatchSize = labels.size(0)

        totalLoss += (
            loss.item()
            * currentBatchSize
        )

        totalCorrect += calculateCorrectPredictions(
            predictions,
            labels,
        )

        totalSamples += currentBatchSize

    averageLoss = totalLoss / totalSamples
    accuracy = totalCorrect / totalSamples

    return averageLoss, accuracy

# Validate
def validateModel(model, dataLoader, lossFunction, device):
    model.eval()

    totalLoss = 0.0
    totalCorrect = 0
    totalSamples = 0

    allLabels = []
    allPredictions = []

    with torch.no_grad():
        for images, labels in dataLoader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            loss = lossFunction(
                predictions,
                labels,
            )

            predictedClasses = predictions.argmax(
                dim=1,
            )

            currentBatchSize = labels.size(0)

            totalLoss += (
                loss.item()
                * currentBatchSize
            )

            totalCorrect += (
                predictedClasses == labels
            ).sum().item()

            totalSamples += currentBatchSize

            allLabels.extend(
                labels.cpu().tolist()
            )

            allPredictions.extend(
                predictedClasses.cpu().tolist()
            )

    averageLoss = totalLoss / totalSamples
    accuracy = totalCorrect / totalSamples

    macroF1 = f1_score(
        allLabels,
        allPredictions,
        average="macro",
        zero_division=0,
    )

    return averageLoss, accuracy, macroF1

validationDataset = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="validation",
    transform=transform,
)

validationLoader = DataLoader(
    dataset=validationDataset,
    batch_size=batchSize,
    shuffle=False,
    num_workers=0,
)


numberOfEpochs = 3

for epochIndex in range(numberOfEpochs):
    trainLoss, trainAccuracy = trainOneEpoch(
        model=model,
        dataLoader=trainLoader,
        lossFunction=lossFunction,
        optimizer=optimizer,
        device=device,
    )

    (
        validationLoss,
        validationAccuracy,
        validationMacroF1,
    ) = validateModel(
        model=model,
        dataLoader=validationLoader,
        lossFunction=lossFunction,
        device=device,
    )

    print(
        f"\nEpoch "
        f"{epochIndex + 1}/{numberOfEpochs}"
    )

    print(
        f"Train loss: {trainLoss:.4f} | "
        f"Train accuracy: {trainAccuracy:.4f}"
    )

    print(
        f"Validation loss: "
        f"{validationLoss:.4f} | "
        f"Validation accuracy: "
        f"{validationAccuracy:.4f} | "
        f"Validation macro F1: "
        f"{validationMacroF1:.4f}"
    )

torch.save(
        model.state_dict(),
        "models/best_efficientNet.pth",
    )