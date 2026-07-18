from pathlib import Path

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
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

checkpointPath = (
    projectRoot
    / "models"
    / "best_season_model.pth"
)

batchSize = 32

classNames = [
    "spring",
    "summer",
    "autumn",
    "winter",
]


def loadSelectedModel(
    checkpointPath,
    device,
):
    if not checkpointPath.is_file():
        raise FileNotFoundError(
            "Selected checkpoint was not found:\n"
            f"{checkpointPath}"
        )

    model, weights = loadPretrainedModel()

    checkpoint = torch.load(
        checkpointPath,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["modelState"]
    )

    model = model.to(device)
    model.eval()

    return model, weights, checkpoint


def evaluateTestSet(
    model,
    dataLoader,
    lossFunction,
    device,
):
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
                dim=1
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

    testLoss = totalLoss / totalSamples
    testAccuracy = totalCorrect / totalSamples

    testMacroF1 = f1_score(
        allLabels,
        allPredictions,
        average="macro",
        zero_division=0,
    )

    return {
        "loss": testLoss,
        "accuracy": testAccuracy,
        "macroF1": testMacroF1,
        "labels": allLabels,
        "predictions": allPredictions,
    }


def main():
    device = getDevice()

    model, weights, checkpoint = (
        loadSelectedModel(
            checkpointPath=checkpointPath,
            device=device,
        )
    )

    transform = weights.transforms()

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

    lossFunction = nn.CrossEntropyLoss()

    results = evaluateTestSet(
        model=model,
        dataLoader=testLoader,
        lossFunction=lossFunction,
        device=device,
    )

    print(f"Device: {device}")
    print(f"Model type: {checkpoint['modelType']}")
    print(f"Test samples: {len(testDataset)}")

    print(
        f"\nTest loss: "
        f"{results['loss']:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Test macro F1: "
        f"{results['macroF1']:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            results["labels"],
            results["predictions"],
            target_names=classNames,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            results["labels"],
            results["predictions"],
        )
    )


main()