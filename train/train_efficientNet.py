from itertools import chain

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from training_helpers import classifierCheckpointPath, createLossFunctions, model, trainEpochs

numberOfEpochs = 20
learningRate = 0.001

seasonLossFunction, subClassLossFunction = createLossFunctions()

classificationParameters = chain(
    model.classifier.parameters(),
    model.seasonHead.parameters(),
    model.subClassHead.parameters(),
)

optimizer = AdamW(
    classificationParameters,
    lr=learningRate,
    weight_decay=1e-4,
)
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=numberOfEpochs,
    eta_min=1e-5,
)

bestAccuracy = trainEpochs(
    numberOfEpochs=numberOfEpochs,
    seasonLossFunction=seasonLossFunction,
    subClassLossFunction=subClassLossFunction,
    optimizer=optimizer,
    model=model,
    destinationPath=classifierCheckpointPath,
    scheduler=scheduler,
    patience=7,
    auxiliaryLossWeight=0.35,
    exponentialMovingAverageDecay=0.995,
    freezeBatchNormalization=True,
    stageName="classifier",
)

print(f"Best CelebA validation accuracy: {bestAccuracy:.4f}")
