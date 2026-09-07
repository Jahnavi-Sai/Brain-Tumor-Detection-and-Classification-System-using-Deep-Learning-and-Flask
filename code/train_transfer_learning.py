import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("Processed_Training", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 3. MobileNetV2 Transfer Learning (Much smaller and faster)
model = models.mobilenet_v2(pretrained=True)

# Freeze layers
for param in model.parameters():
    param.requires_grad = False

# Modify classifier
model.classifier[1] = nn.Sequential(
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 3)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 4. Training
epochs = 25
history = {'loss': [], 'acc': []}

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

# 5. Save (Keep the same filename for simplicity in app.py or update it)
torch.save(model.state_dict(), "brain_tumor_mobilenet.pth")

# 6. Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['acc'], label='Train Accuracy')
plt.title('MobileNetV2 Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.title('MobileNetV2 Loss')
plt.savefig('mobilenet_training_plot.png')
plt.show()
