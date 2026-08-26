import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from training_helpers import checkpointPath, classifierCheckpointPath, createLossFunctions, device, model, trainEpochs

numberOfEpochs = 50
earlyBackboneLearningRate = 3e-6
middleBackboneLearningRate = 1e-5
lateBackboneLearningRate = 3e-5
classifierLearningRate = 1.5e-4


def loadCheckpoint(model, checkpointPath, device):
    if not checkpointPath.is_file():
        raise FileNotFoundError(
            "Classifier checkpoint was not found:\n"
            f"{checkpointPath}\n"
            "Run train/train_efficientNet.py first."
        )

    checkpoint = torch.load(
        checkpointPath,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "modelState" in checkpoint:
        modelState = checkpoint["modelState"]
        modelName = checkpoint.get("modelName")

        if modelName not in {None, "EfficientNetV2SColorAuxiliary"}:
            raise ValueError(
                f"Checkpoint uses the old architecture ({modelName}). "
                "Run train/train_efficientNet.py again before fine-tuning."
            )
    else:
        modelState = checkpoint

    try:
        model.load_state_dict(modelState)
    except RuntimeError as error:
        raise RuntimeError(
            "The classifier checkpoint is incompatible with the improved "
            "model. Run train/train_efficientNet.py again before fine-tuning."
        ) from error

    return model


def unfreezeBackbone(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = True

    return model


def printTrainableParameters(model):
    trainableParameterCount = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Number of trainable scalar parameters: "
        f"{trainableParameterCount:,}"
    )


model = loadCheckpoint(
    model=model,
    checkpointPath=classifierCheckpointPath,
    device=device,
)
model = unfreezeBackbone(model)

printTrainableParameters(model)

seasonLossFunction, subClassLossFunction = createLossFunctions()

optimizer = AdamW(
    [
        {
            "params": model.features[:3].parameters(),
            "lr": earlyBackboneLearningRate,
        },
        {
            "params": model.features[3:6].parameters(),
            "lr": middleBackboneLearningRate,
        },
        {
            "params": model.features[6:].parameters(),
            "lr": lateBackboneLearningRate,
        },
        {
            "params": model.classifier.parameters(),
            "lr": classifierLearningRate,
        },
        {
            "params": model.seasonHead.parameters(),
            "lr": classifierLearningRate,
        },
        {
            "params": model.subClassHead.parameters(),
            "lr": classifierLearningRate,
        },
    ],
    weight_decay=1e-4,
)

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6,
)

bestAccuracy = trainEpochs(
    numberOfEpochs=numberOfEpochs,
    seasonLossFunction=seasonLossFunction,
    subClassLossFunction=subClassLossFunction,
    optimizer=optimizer,
    model=model,
    destinationPath=checkpointPath,
    scheduler=scheduler,
    patience=12,
    auxiliaryLossWeight=0.35,
    exponentialMovingAverageDecay=0.999,
    freezeBatchNormalization=True,
    stageName="fullFineTune",
)

print(f"Best CelebA validation accuracy: {bestAccuracy:.4f}")
