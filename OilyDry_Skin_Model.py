import numpy as np
import torch
from torch import device
import pandas as pd
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import matplotlib.pyplot as plt
from time import time
import cv2
import os



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 3)
)
model = model.to(device)

for param in model.features.parameters():
    param.requires_grad = True

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Skin Dataset\Oily_Dry\Oily-Dry-Skin-Types\train', transform=train_transform)
test_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Skin Dataset\Oily_Dry\Oily-Dry-Skin-Types\test', transform = test_transform)
valid_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Skin Dataset\Oily_Dry\Oily-Dry-Skin-Types\valid', transform= test_transform)

class_names = train_ds.classes

print('Classes:', train_ds.classes)
print('Class mapping:', train_ds.class_to_idx)
print('Number of training samples:', len(train_ds))
print('Number of validation samples:', len(valid_ds))
print('Number of test samples:', len(test_ds))

train_loader = DataLoader(train_ds, batch_size = 32, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size = 32, shuffle=False, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size = 32, shuffle = False, num_workers=0, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 5e-4,
    weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 20
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
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss_sum / len(valid_loader)
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
        _, preds = torch.max(outputs, 1)
        return class_names[preds.item()]