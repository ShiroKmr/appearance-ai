import torch
from pathlib import Path
import torch.optim as optim
import torch.nn as nn

from app.dataset import SeasonDataset
from torch.utils.data import DataLoader

from resnet import testTransform, trainTransform, ResNet18

# Use GPU on Mac
device = torch.device("mps")

# Load the dataset (using standard mean and std for ImageNet
annotationsPath = "assets/deep_armocromia/release/annotations_english.csv"
releaseRoot = "assets/deep_armocromia/release"

testSet = SeasonDataset(
    annotationsPath,
    releaseRoot,
    "test",
    testTransform
)

trainSet = SeasonDataset(
    annotationsPath=annotationsPath,
    releaseRoot=releaseRoot,
    partition="train",
    transform=trainTransform
)

trainLoader = DataLoader(trainSet,batch_size=32,shuffle=True)
testLoader = DataLoader(testSet,batch_size=32,shuffle=False)

model = ResNet18().to(device)
print(model)

# Loss-function and optimizer: cross-entropy and SGD with momentum. Scheduler stepLR slowly decreases learning rate.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Training and testing
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    classCorrect, subClassCorrect, total = 0, 0, 0
    for inputs, classLabels, subClassLabels in trainLoader:
        inputs, classLabels, subClassLabels = inputs.to(device), classLabels.to(device), subClassLabels.to(device)

        optimizer.zero_grad()
        classOutputs, subClassOutputs = model(inputs)

        loss = criterion(classOutputs, classLabels) + 0.5 * criterion(subClassOutputs, subClassLabels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

        _, predictedClasses = classOutputs.max(1)
        _, predictedSubClasses = subClassOutputs.max(1)

        total += classLabels.size(0)

        classCorrect += predictedClasses.eq(classLabels).sum().item()
        subClassCorrect += predictedSubClasses.eq(subClassLabels).sum().item()
    
    train_loss = running_loss / len(trainLoader.dataset)
    trainClassAcc = 100. * classCorrect / total
    trainSubClassAcc = 100. * subClassCorrect / total 

    model.eval()
    classCorrect, subClassCorrect, total = 0, 0, 0

    for inputs, classLabels, subClassLabels in testLoader:
        inputs, classLabels, subClassLabels = inputs.to(device), classLabels.to(device), subClassLabels.to(device)

        classOutputs, subClassOutputs = model(inputs)

        predictedClasses = classOutputs.argmax(dim=1)
        predictedSubClasses = subClassOutputs.argmax(dim=1)

        classCorrect += (predictedClasses == classLabels).sum().item()
        subClassCorrect += (predictedSubClasses == subClassLabels).sum().item()

        total += classLabels.size(0)

    classAccuracy = 100 * classCorrect / total
    subClassAccuracy = 100 * subClassCorrect / total
    
    scheduler.step()
    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Class/Subclass Train Acc: {trainClassAcc:.2f}%, {trainSubClassAcc:.2f}% | Class/Subclass Acc: {classAccuracy:.2f}%, {subClassAccuracy:.2f}%')

modelsPath = Path("models")
modelsPath.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), modelsPath / "resNet18_season.pth")