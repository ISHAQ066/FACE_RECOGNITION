import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np

# Defining the face recognition model using MobileNetV2
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = mobilenet_v2(pretrained=True)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Hyperparameters setting
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 2  # Modify based on the specific task
subset_size = 1000  # Limit the number of samples

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading CelebA dataset
full_train_dataset = CelebA(root='data', split='train', transform=transform, download=True)
full_val_dataset = CelebA(root='data', split='valid', transform=transform, download=True)
full_test_dataset = CelebA(root='data', split='test', transform=transform, download=True)

# Create subsets to run the model faster with limited dataset
train_indices = list(range(subset_size))
val_indices = list(range(subset_size))
test_indices = list(range(subset_size))

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)
test_dataset = Subset(full_test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initializing model, loss function, and optimizer
model = FaceRecognitionModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses, val_losses = [], []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Validation loop with GPU support
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}')

# Visualizing training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()

# Evaluating the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# Saving the model
torch.save(model.state_dict(), 'face_recognition_model_gpu.pth')

# Test the model with a few images
def imshow(img):
    img = img / 2 + 0.5  
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Getting a batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images, labels = images.to(device), labels.to(device)  # Move data to GPU

# Printing images and labels
imshow(torchvision.utils.make_grid(images.cpu()))  # Move images to CPU for display
print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(batch_size)))

# Predicting the labels
outputs = model(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(batch_size)))
