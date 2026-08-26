import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from training_helpers import testLoader, model, validateModel, device, checkpointPath

classNames = ["spring", "summer", "autumn", "winter"]

lossFunction = nn.CrossEntropyLoss()

def loadCheckpoint(model, checkpointPath, device):
    if not checkpointPath.is_file():
        raise FileNotFoundError(
            "Classifier checkpoint was not found:\n"
            f"{checkpointPath}"
        )

    checkpoint = torch.load(
        checkpointPath,
        map_location=device,
        weights_only=False,
    )

    if (isinstance(checkpoint, dict) and "modelState" in checkpoint):
        modelState = checkpoint["modelState"]

        modelName = checkpoint.get("modelName")

        if modelName not in {None, "EfficientNetV2SColorAuxiliary"}:
            raise ValueError(
                f"Checkpoint uses the old architecture ({modelName}). "
                "Retrain both training stages before evaluation."
            )
    else:
        modelState = checkpoint

    try:
        model.load_state_dict(modelState)
    except RuntimeError as error:
        raise RuntimeError(
            "The checkpoint is incompatible with the improved model. "
            "Run train/train_efficientNet.py and "
            "train/tune_efficientNet.py before evaluation."
        ) from error

    return model


model = loadCheckpoint(model, checkpointPath, device)

loss, acc, prec, rec, labels, predictions = validateModel(
    model=model,
    dataLoader=testLoader,
    lossFunction=lossFunction,
    device=device,
    useHorizontalFlipTta=True,
)

print(
    f"\nTest loss: "
    f"{loss}"
)

print(
    f"Test accuracy: "
    f"{acc}"
)

print(
    f"Test macro F1: "
    f"{f1_score(labels, predictions, average='macro', zero_division=0)}"
)

print("\nClassification report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=classNames,
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix:")

print(confusion_matrix(labels, predictions))
