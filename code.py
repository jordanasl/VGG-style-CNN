import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This code runs on google colab with GPU enabled the series of ##### 
 marks the beggining and end of each code cell some cells contain repetitive 
 code due to different code cells in colab that need to be run independently.
"""

# Define CNN Model
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # Block 1: 3 → 32 channels
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 2: 32 → 64 channels
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Block 3: 64 → 128 channels
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        # Block 4: 128 → 256 channels
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)

        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # Block 1 forward
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 2 forward
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        # Block 3 forward
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        # Block 4 forward
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)

        # Flatten feature maps
        x = torch.flatten(x, 1)

        # Fully connected forward
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

######

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Data augmentation for training set
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


batch_size = 64


trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)


testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)


net = cnn().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    net.parameters(), lr=0.001, weight_decay=0.0005
)

#
num_epochs = 100


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)


history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

start = time.time()


for epoch in range(num_epochs):

    net.train()  
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          
        outputs = net(images)          
        loss = criterion(outputs, labels)
        loss.backward()                
        optimizer.step()              

        running_train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    
    current_lr = optimizer.param_groups[0]['lr']
    history['lr'].append(current_lr)

    scheduler.step()  

    net.eval()  
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

   
    epoch_train_loss = running_train_loss / total_train
    epoch_train_acc = 100 * correct_train / total_train
    epoch_val_loss = running_val_loss / total_val
    epoch_val_acc = 100 * correct_val / total_val

    
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {epoch_train_loss:.3f} | Train Acc: {epoch_train_acc:.2f}% | "
        f"Val Loss: {epoch_val_loss:.3f} | Val Acc: {epoch_val_acc:.2f}%"
    )

print(f"Time: {(time.time() - start)/60:.2f} minutes")


PATH = "./cifar_net_trained.pth"
torch.save(net.state_dict(), PATH)
print(f"saved{PATH}")

# Plot training curves
epochs_range = range(1, num_epochs + 1)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, history['train_loss'], label='Training Loss')
plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training / Validation Accuracy")

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['lr'])
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)

plt.tight_layout()
plt.show()

####

# Test phase

correct = 0
total = 0

net.eval()  

with torch.no_grad():  
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Final test accuracy
accuracy = 100 * correct / total
print(f'Accuracy στο Test Set: {accuracy:.2f} %')
