import numpy as np
import torch
from torch import device
import pandas as pd
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
from time import time
import cv2
import splitfolders
import os
from collections import Counter



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False 
for param in model.features[3:].parameters():
    param.requires_grad = True

num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 3)
)
model = model.to(device)

train_transform = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Very mild jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((228, 228)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# input_folder = r'C:\asasd\SkinSync\OilyDrySkin_PreSplit'
output_folder = r'C:\asasd\SkinSync\OilyDrySkin_PostSplit'

# splitfolders.ratio(input=input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1), group_prefix=None, move=False)

train_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Data\OilyDrySkin_PostSplit\train', transform=train_transform)
test_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Data\OilyDrySkin_PostSplit\test', transform = test_transform)
val_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Data\OilyDrySkin_PostSplit\val', transform = test_transform)

class_names = train_ds.classes

print('Classes:', train_ds.classes)
print('Class mapping:', train_ds.class_to_idx)
print('Number of training samples:', len(train_ds))
print('Number of validation samples:', len(val_ds))
print('Number of test samples:', len(test_ds))

print("Samples per class in training:", Counter(train_ds.targets))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, smoothing=0.1):
        super(FocalLossWithSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.smoothing / (num_classes - 1), device=inputs.device)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_preds = nn.functional.log_softmax(inputs, dim=1)
        ce_loss = -torch.sum(smooth_targets * log_preds, dim=1)
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Multiply by class weights (alpha) if provided
        if self.alpha is not None:
            alpha_weights = self.alpha[targets]
            focal_loss = focal_loss * alpha_weights

        return focal_loss.mean()

targets = train_ds.targets
class_counts = Counter(targets)
counts = np.array([class_counts[i] for i in range(len(train_ds.classes))])
smoothed_weights = 1.0 / (counts ** (1/3))
smoothed_weights = torch.tensor(smoothed_weights / smoothed_weights.sum(), dtype=torch.float).to(device)

criterion = FocalLossWithSmoothing(alpha=smoothed_weights, gamma=1.5)
optimizer = torch.optim.AdamW([
    {'params': model.features[3:].parameters(), 'lr': 3e-5}, 
    {'params': model.classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-2)

epochs = 25
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
best_val_acc = 0.0
best_model_path = 'best_skin_type_model.pth'

for epoch in range(epochs):
    start_time = time()

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_total += labels.size(0)
        running_correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = running_correct / running_total

    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss_sum / len(val_loader)
    val_acc = val_correct / val_total
    epoch_time = time() - start_time

    print(
        f'''Epoch [{epoch+1}/{epochs}]
        Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}
        Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}
        Time: {epoch_time:.1f}s'''
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'--> New best model saved with val acc: {best_val_acc:.4f}')

    scheduler.step()

model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)

model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

test_acc = test_correct / test_total
print(f'Test Accuracy: {test_acc:.4f}')

def predict_image(img_path, model, transform, class_names):
    model.eval()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = torchvision.transforms.functional.to_pil_image(img)
    img_t = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probabilities, 1)
        
        predicted_class = class_names[preds.item()]
        confidence_pct = conf.item() * 100
        
        return predicted_class, confidence_pct 
    

from sklearn.metrics import classification_report, confusion_matrix

all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n--- Test Set Evaluation ---")
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=class_names))