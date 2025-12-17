"""
 This is the executable file for verifying the model's accuracy
 The dataset is set to download automatically if it does not exist in the folder
 A simple check is included to see whether the code runs on GPU (CUDA) or CPU
 The trained model is loaded from the variable PATH on line 106, this is the only manual change required
 before running.
 After printing the accuracy in the terminal, a window will appear showing all misclassified images.
 Once this window is closed and after <10 seconds, a second window opens showing 10 correctly classified images 
 (one per class)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Define CNN model 
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()

        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)  # Max-pooling layer

        # Fully connected MLP head
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        # Forward pass through convolutional blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool(x)

        # Flatten feature maps for MLP head
        x = torch.flatten(x, 1)

        # Fully connected MLP forward
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Output logits

        return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)  

    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

   
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    # Path to trained model (manual change required)
    PATH = r""

    # Load model
    model = cnn().to(device)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()  

   
    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f} %") 

########3

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import random

# Store misclassified images
mis_images = []
mis_preds = []
mis_labels = []

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        mismatch = predicted != labels  

        if mismatch.any():
            mis_images.append(images[mismatch].cpu())
            mis_preds.append(predicted[mismatch].cpu())
            mis_labels.append(labels[mismatch].cpu())

# Concatenate all misclassified images
mis_images = torch.cat(mis_images)
mis_preds  = torch.cat(mis_preds)
mis_labels = torch.cat(mis_labels)

print(f"Number of misclassified images: {len(mis_images)}")

# Visualize misclassified images with next/prev buttons
index = 0
fig, ax = plt.subplots(figsize=(4,4))
plt.subplots_adjust(bottom=0.2)

def show_image(i):
    ax.clear()
    img = mis_images[i] / 2 + 0.5  
    npimg = img.numpy().transpose(1, 2, 0)

    ax.imshow(npimg)
    ax.set_title(f"Predicted: {classes[mis_preds[i].item()]} | True: {classes[mis_labels[i].item()]}")
    ax.axis('off')
    fig.canvas.draw()

show_image(index)

prev_ax = plt.axes([0.2, 0.05, 0.2, 0.075])
btn_prev = Button(prev_ax, '<-')

def prev(event):
    global index
    index = (index - 1) % len(mis_images)
    show_image(index)

btn_prev.on_clicked(prev)

next_ax = plt.axes([0.6, 0.05, 0.2, 0.075])
btn_next = Button(next_ax, '->')

def next(event):
    global index
    index = (index + 1) % len(mis_images)
    show_image(index)

btn_next.on_clicked(next)

plt.show()

# Select one correct image per class
all_correct_images = {i: [] for i in range(10)}  

model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        matches = predicted == labels

        for img, pred, ok in zip(images, predicted, matches):
            if ok:
                cls = pred.item()
                all_correct_images[cls].append(img.cpu())

selected_images = []
selected_classes = []

for cls in range(10):
    if len(all_correct_images[cls]) > 0:
        idx = random.randint(0, len(all_correct_images[cls]) - 1)
        selected_images.append(all_correct_images[cls][idx])
        selected_classes.append(classes[cls])

# Plot one correct image per class
plt.figure(figsize=(12, 6))
for i in range(10):
    img = selected_images[i]
    img = img / 2 + 0.5  
    npimg = img.numpy().transpose(1, 2, 0)

    plt.subplot(2, 5, i + 1)
    plt.imshow(npimg)
    plt.title(f"{selected_classes[i]}")
    plt.axis("off")

plt.suptitle("10 random correctly classified images – one per class", fontsize=16)
plt.tight_layout()
plt.show()
