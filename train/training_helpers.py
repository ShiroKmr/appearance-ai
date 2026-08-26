import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset import SeasonDataset, classToIndex, subClassToIndex
from model_efficientNet import getDevice, loadPretrainedModel


projectRoot = Path(__file__).resolve().parents[1]

releaseRoot = (
    projectRoot
    / "assets"
    / "deep_armocromia"
    / "release"
)
annotationsPath = releaseRoot / "annotations_with_validation.csv"
modelsRoot = projectRoot / "models"
classifierCheckpointPath = modelsRoot / "best_efficientNet.pth"
checkpointPath = modelsRoot / "best_tune_efficientNet.pth"

randomSeed = 42
trainingBatchSize = 8
evaluationBatchSize = 16
sourceMatchWeight = 2.0


def setRandomSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


setRandomSeed(randomSeed)

device = getDevice()
model, weights = loadPretrainedModel()
model = model.to(device)

weightTransform = weights.transforms()
imageSize = weightTransform.crop_size[0]

# Season labels depend on colour. Geometry is augmented, while hue, saturation,
# white balance, and contrast are deliberately left untouched.
trainTransform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=imageSize,
            scale=(0.82, 1.0),
            ratio=(0.85, 1.15),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [
                transforms.RandomRotation(
                    degrees=6,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
            ],
            p=0.35,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weightTransform.mean,
            std=weightTransform.std,
        ),
    ]
)
evaluationTransform = weightTransform

trainDataset = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="train",
    transform=trainTransform,
    includeSubClass=True,
    includePivotalValidation=True,
)

validationDataset = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="validation",
    transform=evaluationTransform,
    celebaOnly=True,
)

testDataset = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="test",
    transform=evaluationTransform,
)

samplingGenerator = torch.Generator()
samplingGenerator.manual_seed(randomSeed)

trainSampler = WeightedRandomSampler(
    weights=trainDataset.getSourceSamplingWeights(
        celebaWeight=sourceMatchWeight
    ),
    num_samples=len(trainDataset),
    replacement=True,
    generator=samplingGenerator,
)

loaderArguments = {
    "num_workers": 0,
    "pin_memory": device.type == "cuda",
}

trainLoader = DataLoader(
    dataset=trainDataset,
    batch_size=trainingBatchSize,
    sampler=trainSampler,
    **loaderArguments,
)
validationLoader = DataLoader(
    dataset=validationDataset,
    batch_size=evaluationBatchSize,
    shuffle=False,
    **loaderArguments,
)
testLoader = DataLoader(
    dataset=testDataset,
    batch_size=evaluationBatchSize,
    shuffle=False,
    **loaderArguments,
)


def calculateBalancedWeights(dataSet, columnName, valueToIndex):
    valueCounts = dataSet.annotations[columnName].value_counts()
    sampleCount = len(dataSet)
    valueCount = len(valueToIndex)

    balancedWeights = torch.empty(valueCount, dtype=torch.float32)

    for valueName, valueIndex in valueToIndex.items():
        exactBalancedWeight = (
            sampleCount
            / (valueCount * int(valueCounts[valueName]))
        )
        balancedWeights[valueIndex] = exactBalancedWeight ** 0.5

    return balancedWeights / balancedWeights.mean()


seasonClassWeights = calculateBalancedWeights(
    trainDataset,
    "class",
    classToIndex,
)
subClassWeights = calculateBalancedWeights(
    trainDataset,
    "sub_class",
    subClassToIndex,
)


def createLossFunctions():
    seasonLossFunction = nn.CrossEntropyLoss(
        weight=seasonClassWeights.to(device),
        label_smoothing=0.05,
    )
    subClassLossFunction = nn.CrossEntropyLoss(
        weight=subClassWeights.to(device),
        label_smoothing=0.03,
    )

    return seasonLossFunction, subClassLossFunction


def setBatchNormalizationEvaluation(modelToUpdate):
    for module in modelToUpdate.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def calculateCorrectPredictions(predictions, labels):
    predictedClasses = predictions.argmax(dim=1)

    return (predictedClasses == labels).sum().item()


def trainOneEpoch(
    model,
    dataLoader,
    seasonLossFunction,
    subClassLossFunction,
    optimizer,
    device,
    auxiliaryLossWeight=0.35,
    gradientClipNorm=1.0,
    gradientAccumulationSteps=2,
    exponentialMovingAverage=None,
    freezeBatchNormalization=True,
):
    model.train()

    if freezeBatchNormalization:
        setBatchNormalizationEvaluation(model)

    totalLoss = 0.0
    totalCorrect = 0
    totalSamples = 0

    optimizer.zero_grad(set_to_none=True)

    for batchIndex, (images, labels, subClassLabels) in enumerate(dataLoader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        subClassLabels = subClassLabels.to(device, non_blocking=True)

        predictions, subClassPredictions = model.forwardWithAuxiliary(images)
        seasonLoss = seasonLossFunction(predictions, labels)
        auxiliaryLoss = subClassLossFunction(
            subClassPredictions,
            subClassLabels,
        )
        loss = seasonLoss + auxiliaryLossWeight * auxiliaryLoss

        (loss / gradientAccumulationSteps).backward()

        shouldUpdate = (
            (batchIndex + 1) % gradientAccumulationSteps == 0
            or batchIndex + 1 == len(dataLoader)
        )

        if shouldUpdate:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradientClipNorm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if exponentialMovingAverage is not None:
                exponentialMovingAverage.update_parameters(model)

        currentBatchSize = labels.size(0)
        totalLoss += loss.item() * currentBatchSize
        totalCorrect += calculateCorrectPredictions(predictions, labels)
        totalSamples += currentBatchSize

    averageLoss = totalLoss / totalSamples
    accuracy = totalCorrect / totalSamples

    return averageLoss, accuracy


def validateModel(
    model,
    dataLoader,
    lossFunction,
    device,
    useHorizontalFlipTta=False,
):
    model.eval()

    totalLoss = 0.0
    totalCorrect = 0
    totalSamples = 0

    allLabels = []
    allPredictions = []

    with torch.inference_mode():
        for batch in dataLoader:
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(images)

            if useHorizontalFlipTta:
                flippedPredictions = model(torch.flip(images, dims=[3]))
                predictions = (predictions + flippedPredictions) / 2.0

            loss = lossFunction(predictions, labels)
            predictedClasses = predictions.argmax(dim=1)
            currentBatchSize = labels.size(0)

            totalLoss += loss.item() * currentBatchSize
            totalCorrect += (predictedClasses == labels).sum().item()
            totalSamples += currentBatchSize

            allLabels.extend(labels.cpu().tolist())
            allPredictions.extend(predictedClasses.cpu().tolist())

    averageLoss = totalLoss / totalSamples
    accuracy = totalCorrect / totalSamples
    precision = precision_score(
        allLabels,
        allPredictions,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        allLabels,
        allPredictions,
        average="macro",
        zero_division=0,
    )

    return (
        averageLoss,
        accuracy,
        precision,
        recall,
        allLabels,
        allPredictions,
    )


def saveCheckpoint(
    model,
    validationMetrics,
    epochIndex,
    destinationPath,
    stageName,
):
    destinationPath.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "modelState": model.state_dict(),
        "epoch": epochIndex + 1,
        "validationAccuracy": validationMetrics["accuracy"],
        "validationMacroF1": validationMetrics["macroF1"],
        "validationLoss": validationMetrics["loss"],
        "classToIndex": classToIndex,
        "subClassToIndex": subClassToIndex,
        "modelName": "EfficientNetV2SColorAuxiliary",
        "imageSize": imageSize,
        "trainingStage": stageName,
        "validationDomain": "CelebA",
    }

    torch.save(checkpoint, destinationPath)


def trainEpochs(
    numberOfEpochs,
    seasonLossFunction,
    subClassLossFunction,
    optimizer,
    model,
    destinationPath,
    scheduler=None,
    patience=10,
    auxiliaryLossWeight=0.35,
    gradientAccumulationSteps=2,
    exponentialMovingAverageDecay=0.999,
    freezeBatchNormalization=True,
    stageName="fineTune",
):
    bestAccuracy = -1.0
    bestMacroF1 = -1.0
    epochsWithoutImprovement = 0

    exponentialMovingAverage = None

    if exponentialMovingAverageDecay is not None:
        exponentialMovingAverage = AveragedModel(
            model,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(
                exponentialMovingAverageDecay
            ),
            use_buffers=True,
        )

    for epochIndex in range(numberOfEpochs):
        trainLoss, trainAccuracy = trainOneEpoch(
            model=model,
            dataLoader=trainLoader,
            seasonLossFunction=seasonLossFunction,
            subClassLossFunction=subClassLossFunction,
            optimizer=optimizer,
            device=device,
            auxiliaryLossWeight=auxiliaryLossWeight,
            gradientAccumulationSteps=gradientAccumulationSteps,
            exponentialMovingAverage=exponentialMovingAverage,
            freezeBatchNormalization=freezeBatchNormalization,
        )

        evaluationModel = (
            exponentialMovingAverage
            if exponentialMovingAverage is not None
            else model
        )

        (
            validationLoss,
            validationAccuracy,
            validationPrecision,
            validationRecall,
            validationLabels,
            validationPredictions,
        ) = validateModel(
            model=evaluationModel,
            dataLoader=validationLoader,
            lossFunction=seasonLossFunction,
            device=device,
        )

        validationMacroF1 = f1_score(
            validationLabels,
            validationPredictions,
            average="macro",
            zero_division=0,
        )

        learningRates = [
            parameterGroup["lr"]
            for parameterGroup in optimizer.param_groups
        ]

        print(f"\nEpoch {epochIndex + 1}/{numberOfEpochs}")
        print(
            f"Train loss: {trainLoss:.4f} | "
            f"Train accuracy: {trainAccuracy:.4f}"
        )
        print(
            f"Validation loss: {validationLoss:.4f} | "
            f"Validation accuracy: {validationAccuracy:.4f} | "
            f"Validation macro F1: {validationMacroF1:.4f} | "
            f"Precision: {validationPrecision:.4f} | "
            f"Recall: {validationRecall:.4f}"
        )
        print(
            "Learning rates: "
            + ", ".join(f"{rate:.2e}" for rate in learningRates)
        )

        accuracyImproved = validationAccuracy > bestAccuracy
        f1BrokeTie = (
            validationAccuracy == bestAccuracy
            and validationMacroF1 > bestMacroF1
        )

        if accuracyImproved or f1BrokeTie:
            bestAccuracy = validationAccuracy
            bestMacroF1 = validationMacroF1
            epochsWithoutImprovement = 0

            modelToSave = (
                exponentialMovingAverage.module
                if exponentialMovingAverage is not None
                else model
            )
            validationMetrics = {
                "loss": validationLoss,
                "accuracy": validationAccuracy,
                "macroF1": validationMacroF1,
            }
            saveCheckpoint(
                model=modelToSave,
                validationMetrics=validationMetrics,
                epochIndex=epochIndex,
                destinationPath=destinationPath,
                stageName=stageName,
            )
            print(f"Saved the best model to {destinationPath}")
        else:
            epochsWithoutImprovement += 1

        if scheduler is not None:
            scheduler.step()

        if epochsWithoutImprovement >= patience:
            print("Early stopping")
            break

    return bestAccuracy
