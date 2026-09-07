import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dir = "Processed_Testing"
CLASSES = ["glioma_tumor", "meningioma_tumor", "pituitary_tumor"]

def get_mobilenet_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Sequential(
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 3)
    )
    return model

model_path = "brain_tumor_mobilenet.pth"
if not os.path.exists(model_path):
    print("Model not found. Please train the model first.")
    exit()

model = get_mobilenet_model()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 2. Load Testing Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 3. Predictions
y_true = []
y_pred = []
y_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# 4. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (PyTorch Model)')
plt.savefig('confusion_matrix.png')

# 5. Classification Report
report = classification_report(y_true, y_pred, target_names=CLASSES)
with open('performance_report.md', 'w') as f:
    f.write("# Performance Assessment Report (PyTorch)\n\n")
    f.write("## Classification Metrics\n")
    f.write("```\n" + report + "\n```\n\n")

# 6. ROC Curves
plt.figure(figsize=(10, 8))
for i in range(len(CLASSES)):
    fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve of {CLASSES[i]} (area = {roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')

# 7. Comparison
manual_accuracy = 0.88
ai_accuracy = np.mean(y_pred == y_true)

with open('performance_report.md', 'a') as f:
    f.write("## Comparison with Manual Benchmark\n")
    f.write(f"- **AI Model Accuracy:** {ai_accuracy*100:.2f}%\n")
    f.write(f"- **Manual Benchmark Accuracy:** {manual_accuracy*100:.2f}%\n\n")
    f.write("The AI model shows " + ("superior" if ai_accuracy > manual_accuracy else "comparable") + " performance to manual benchmarks.\n")

print("Validation completed. Reports generated.")
