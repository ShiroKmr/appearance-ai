import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from pathlib import Path

from app.dataset import SeasonDataset
from torch.utils.data import DataLoader

# Use GPU on Mac
device = torch.device("mps")

# Load the dataset (using standard mean and std for ImageNet
annotationsPath = "assets/deep_armocromia/release/annotations_english.csv"
releaseRoot = "assets/deep_armocromia/release"

trainTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

testTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

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

# Add the residual block: 2 convolution, batchNorm and ReLu
class BasicBlock(nn.Module):
    def __init__(self, inChannels, outChannels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(outChannels,outChannels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.shortcut = nn.Sequential()

        if stride != 1 or inChannels != outChannels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannels,outChannels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outChannels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Define the structure:
# 4 layers of residual blocks with increasing channels
# Each layer consists of multiple residual blocks
# Global Average Pooling averaging over feature maps, converting them to a vector
# Fully Connected Layer outputs prediction for 12 classes
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxPool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.classHead = nn.Linear(512, 4)
        self.subClassHead = nn.Linear(512, 6)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxPool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgPool(out)
        out = torch.flatten(out, 1)

        classOut = self.classHead(out)
        subClassOut = self.subClassHead(out)

        return classOut, subClassOut

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