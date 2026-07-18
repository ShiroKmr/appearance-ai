# Fine-tune the model
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import SeasonDataset
from model_efficientNet import (
    getDevice,
    loadPretrainedModel,
)


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

fineTunedCheckpointPath = (
    modelsRoot
    / "best_fine_tuned.pth"
)


batchSize = 32
numberOfEpochs = 30
patience = 5

backboneLearningRate = 0.00001
classifierLearningRate = 0.0001


def loadCheckpoint(model, checkpointPath, device):
    if not checkpointPath.is_file():
        raise FileNotFoundError(
            "Classifier checkpoint was not found:\n"
            f"{checkpointPath}"
        )

    checkpoint = torch.load(
        checkpointPath,
        map_location=device,
    )

    if (isinstance(checkpoint, dict) and "modelState" in checkpoint):
        modelState = checkpoint["modelState"]
    else:
        modelState = checkpoint

    model.load_state_dict(modelState)

    return model


def unfreezeLastBlocks(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

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
            print(f"  {parameterName}")
            trainableParameterCount += parameter.numel()

    print(
        "\nNumber of trainable scalar parameters: "
        f"{trainableParameterCount}"
    )


def calculateCorrectPredictions(predictions, labels):
    predictedClasses = predictions.argmax(dim=1)

    correctPredictions = (
        predictedClasses == labels
    ).sum().item()

    return correctPredictions


def trainOneEpoch(model, dataLoader, lossFunction, optimizer, device):
    model.train()

    for blockIndex in range(7):
        model.features[blockIndex].eval()

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


def saveCheckpoint(model, validationMacroF1, epochIndex, checkpointPath):
    checkpoint = {
        "modelState": model.state_dict(),
        "validationMacroF1": validationMacroF1,
        "epoch": epochIndex + 1,
        "classToIndex": {
            "spring": 0,
            "summer": 1,
            "autumn": 2,
            "winter": 3,
        },
        "modelName": "EfficientNetB0",
        "fineTunedBlocks": [7, 8],
    }

    torch.save(
        checkpoint,
        checkpointPath,
    )


def main():
    modelsRoot.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not annotationsPath.is_file():
        raise FileNotFoundError(
            "Annotations file was not found:\n"
            f"{annotationsPath}"
        )

    device = getDevice()

    model, weights = loadPretrainedModel()

    model = loadCheckpoint(
        model=model,
        checkpointPath=classifierCheckpointPath,
        device=device,
    )

    model = unfreezeLastBlocks(model)
    model = model.to(device)

    print(f"Device: {device}")
    print(
        "Loaded classifier checkpoint: "
        f"{classifierCheckpointPath}"
    )

    printTrainableParameters(model)

    transform = weights.transforms()

    trainDataset = SeasonDataset(
        annotationsPath=annotationsPath,
        releaseRoot=releaseRoot,
        partition="train",
        transform=transform,
    )

    validationDataset = SeasonDataset(
        annotationsPath=annotationsPath,
        releaseRoot=releaseRoot,
        partition="validation",
        transform=transform,
    )

    trainLoader = DataLoader(
        dataset=trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
    )

    validationLoader = DataLoader(
        dataset=validationDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
    )

    lossFunction = nn.CrossEntropyLoss()

    optimizer = Adam(
        [
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

    bestValidationMacroF1 = 0.0
    epochsWithoutImprovement = 0

    print(f"Train samples: {len(trainDataset)}")
    print(
        "Validation samples: "
        f"{len(validationDataset)}"
    )

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
            f"\nFine-tuning epoch "
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

        if (
            validationMacroF1
            > bestValidationMacroF1
        ):
            bestValidationMacroF1 = (
                validationMacroF1
            )

            epochsWithoutImprovement = 0

            saveCheckpoint(
                model=model,
                validationMacroF1=validationMacroF1,
                epochIndex=epochIndex,
                checkpointPath=fineTunedCheckpointPath,
            )

            print(
                "New best fine-tuned checkpoint saved:"
            )
            print(f"  {fineTunedCheckpointPath}")

        else:
            epochsWithoutImprovement += 1

            print(
                "No validation macro F1 improvement "
                f"for {epochsWithoutImprovement} "
                "epoch(s)."
            )

        if (
            epochsWithoutImprovement
            >= patience
        ):
            print("\nEarly stopping.")
            break

    print(
        "\nBest validation macro F1: "
        f"{bestValidationMacroF1:.4f}"
    )


main()