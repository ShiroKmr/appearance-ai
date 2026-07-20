# Train the model
from model_efficientNet import getDevice, loadPretrainedModel

# Work with a dataset
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import SeasonDataset

# Validation
import torch
from sklearn.metrics import precision_score, recall_score

# Define the paths
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

modelsRoot = projectRoot / "models"

classifierCheckpointPath = (
    modelsRoot
    / "best_efficientNet.pth"
)

checkpointPath = (
    projectRoot
    / "models"
    / "best_tune_efficientNet.pth"
)

# Get the model
device = getDevice()

model, weights = loadPretrainedModel()
model = model.to(device)

# Define the loaders for training/validation/testing
batchSize = 16
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

testDataset = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="test",
    transform=transform,
)

testLoader = DataLoader(
    dataset=testDataset,
    batch_size=batchSize,
    shuffle=False,
    num_workers=0,
)

# Validation and training functions
def calculateCorrectPredictions(predictions, labels):
    predictedClasses = predictions.argmax(dim=1)

    correctPredictions = (predictedClasses == labels).sum().item()

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

        loss = lossFunction(predictions, labels)

        loss.backward()
        optimizer.step()

        currentBatchSize = labels.size(0)

        totalLoss += (loss.item() * currentBatchSize)

        totalCorrect += calculateCorrectPredictions(predictions, labels)

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

            loss = lossFunction(predictions, labels)

            predictedClasses = predictions.argmax(dim=1)

            currentBatchSize = labels.size(0)

            totalLoss += (loss.item() * currentBatchSize)

            totalCorrect += (predictedClasses == labels).sum().item()

            totalSamples += currentBatchSize

            allLabels.extend(labels.cpu().tolist())
            allPredictions.extend(predictedClasses.cpu().tolist())

    averageLoss = totalLoss / totalSamples
    accuracy = totalCorrect / totalSamples

    prec = precision_score(allLabels, allPredictions, average="macro", zero_division=0)
    rec = recall_score(allLabels, allPredictions, average="macro", zero_division=0)

    return averageLoss, accuracy, prec, rec, allLabels, allPredictions

# Training

def saveCheckpoint(model, prec, rec, epochIndex, checkpointPath):
    checkpoint = {
        "modelState": model.state_dict(),
        "Prec": prec,
        "Rec": rec,
        "epoch": epochIndex + 1,
        "classToIndex": {
            "spring": 0,
            "summer": 1,
            "autumn": 2,
            "winter": 3,
        },
        "modelName": "EfficientNetB0",
        "fineTunedBlocks": [5, 6, 7, 8],
    }

    torch.save(checkpoint, checkpointPath)

def trainEpochs(epoch, lossFunction, optimizer, model, tune):
    patience = 5
    bestAcc = 0.0
    for epochIndex in range(epoch):
        trainLoss, trainAccuracy = trainOneEpoch(
            model=model,
            dataLoader=trainLoader,
            lossFunction=lossFunction,
            optimizer=optimizer,
            device=device,
        )

        validationLoss, validationAccuracy, validationPrec, validationRec, _, _ = validateModel(
            model=model,
            dataLoader=validationLoader,
            lossFunction=lossFunction,
            device=device,
        )

        print(
            f"\nEpoch "
            f"{epochIndex + 1}/{epoch}"
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
            f"Validation Precision: "
            f"{validationPrec:.4f} | "
            f"Validation Recall: "
            f"{validationRec:.4f}"
        )

        if patience > 0:
            if (bestAcc < validationAccuracy):
                bestAcc = validationAccuracy
                if tune:
                    saveCheckpoint(model, validationPrec, validationRec, epochIndex, checkpointPath)
                else:
                    torch.save(model.state_dict(), "models/best_efficientNet.pth")
                print("Saved the best model")
                patience = 5
            else:
                patience -= 1
                if patience == 0:
                    print("Early stopping")
                    break
        else:
            break
    
    return bestAcc