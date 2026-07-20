# Train the model
from torch import nn
from torch.optim import Adam

from training_helpers import trainEpochs, model

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
lossFunction = nn.CrossEntropyLoss()
optimizer = Adam(model.classifier.parameters(), lr=learningRate)
numberOfEpochs = 10

tuneIndicator = False

trainEpochs(numberOfEpochs, lossFunction, optimizer, model, tuneIndicator)
