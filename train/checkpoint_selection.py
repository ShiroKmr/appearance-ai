# Choose the best between the trained and fine-tuned model
from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch import nn
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

selectedCheckpointPath = (
    modelsRoot
    / "best_season_model.pth"
)

batchSize = 32


def extractModelState(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "modelState" in checkpoint
    ):
        return checkpoint["modelState"]

    return checkpoint


def loadModelFromCheckpoint(
    checkpointPath,
    device,
):
    if not checkpointPath.is_file():
        raise FileNotFoundError(
            "Checkpoint was not found:\n"
            f"{checkpointPath}"
        )

    model, weights = loadPretrainedModel()

    checkpoint = torch.load(
        checkpointPath,
        map_location=device,
    )

    modelState = extractModelState(checkpoint)

    model.load_state_dict(modelState)
    model = model.to(device)

    return model, weights, checkpoint


def evaluateModel(
    model,
    dataLoader,
    lossFunction,
    device,
):
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

    return {
        "loss": averageLoss,
        "accuracy": accuracy,
        "macroF1": macroF1,
    }


def printMetrics(
    checkpointName,
    metrics,
):
    print(f"\n{checkpointName}")
    print(
        f"Validation loss: "
        f"{metrics['loss']:.4f}"
    )
    print(
        f"Validation accuracy: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Validation macro F1: "
        f"{metrics['macroF1']:.4f}"
    )


def main():
    device = getDevice()

    classifierModel, weights, classifierCheckpoint = (
        loadModelFromCheckpoint(
            checkpointPath=classifierCheckpointPath,
            device=device,
        )
    )

    fineTunedModel, _, fineTunedCheckpoint = (
        loadModelFromCheckpoint(
            checkpointPath=fineTunedCheckpointPath,
            device=device,
        )
    )

    transform = weights.transforms()

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

    lossFunction = nn.CrossEntropyLoss()

    classifierMetrics = evaluateModel(
        model=classifierModel,
        dataLoader=validationLoader,
        lossFunction=lossFunction,
        device=device,
    )

    fineTunedMetrics = evaluateModel(
        model=fineTunedModel,
        dataLoader=validationLoader,
        lossFunction=lossFunction,
        device=device,
    )

    print(f"Device: {device}")
    print(
        f"Validation samples: "
        f"{len(validationDataset)}"
    )

    printMetrics(
        checkpointName="Classifier checkpoint",
        metrics=classifierMetrics,
    )

    printMetrics(
        checkpointName="Fine-tuned checkpoint",
        metrics=fineTunedMetrics,
    )

    if (
        fineTunedMetrics["macroF1"]
        > classifierMetrics["macroF1"]
    ):
        bestCheckpointPath = (
            fineTunedCheckpointPath
        )
        bestCheckpoint = fineTunedCheckpoint
        bestMetrics = fineTunedMetrics
        bestModelType = "fine-tuned"
    else:
        bestCheckpointPath = (
            classifierCheckpointPath
        )
        bestCheckpoint = classifierCheckpoint
        bestMetrics = classifierMetrics
        bestModelType = "classifier-only"

    selectedCheckpoint = {
        "modelState": extractModelState(
            bestCheckpoint
        ),
        "modelType": bestModelType,
        "validationLoss": bestMetrics["loss"],
        "validationAccuracy": (
            bestMetrics["accuracy"]
        ),
        "validationMacroF1": (
            bestMetrics["macroF1"]
        ),
        "classToIndex": {
            "spring": 0,
            "summer": 1,
            "autumn": 2,
            "winter": 3,
        },
        "modelName": "EfficientNetB0",
        "inputSize": [224, 224],
    }

    torch.save(
        selectedCheckpoint,
        selectedCheckpointPath,
    )

    print("\nSelected model:")
    print(f"  Type: {bestModelType}")
    print(
        f"  Validation macro F1: "
        f"{bestMetrics['macroF1']:.4f}"
    )
    print(
        f"  Source checkpoint: "
        f"{bestCheckpointPath}"
    )
    print(
        f"  Saved as: "
        f"{selectedCheckpointPath}"
    )


main()