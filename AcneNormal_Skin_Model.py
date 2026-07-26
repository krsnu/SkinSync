import numpy as np
import torch
from torch import device
import pandas as pd
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from time import time
import cv2
import os
from collections import Counter
import splitfolders




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False 
for i in range(10, 19):
    for param in model.features[i].parameters():
        param.requires_grad = True

num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 2)
)
model = model.to(device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=(0.3, 1.8), contrast=0.4, saturation=0.2),
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

# input_folder = r'C:\asasd\SkinSync\Data\AcneSkinData_PreSplit'
# output_folder = r'C:\asasd\SkinSync\Data\AcneSkin_PostSplit'

# splitfolders.ratio(input=input_folder, output=output_folder, seed=42, ratio=(.8, .1, .1), group_prefix=None, move=False)

train_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Data\AcneSkin_PostSplit\train', transform=train_transform)
test_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Data\AcneSkin_PostSplit\test', transform = test_transform)
val_ds = datasets.ImageFolder(r'C:\asasd\SkinSync\Data\AcneSkin_PostSplit\val', transform = test_transform)

class_names = train_ds.classes

print('Classes:', train_ds.classes)
print('Class mapping:', train_ds.class_to_idx)
print('Number of training samples:', len(train_ds))
print('Number of validation samples:', len(val_ds))
print('Number of test samples:', len(test_ds))

print("Samples per class in training:", Counter(train_ds.targets))

targets = np.array(train_ds.targets)
class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
weight = 1. / class_sample_count
print(f"Calculated class weights: {weight}")

samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight).float()
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, shuffle=False, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

backbone_params = []
for i in range(10, 19):
    backbone_params.extend(list(model.features[i].parameters()))


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4} 
], weight_decay=1e-4)

epochs = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
best_val_acc = 0.0
best_model_path = 'best_acne_detection_model.pth'

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

def predict_image(img_path, model, transform, class_names, acne_threshold = 0.3):
    model.eval()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Image not found: {img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = torchvision.transforms.functional.to_pil_image(img)
    img_t = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probabilities = torch.softmax(outputs, dim=1)[0]

        acne_idx = class_names.index('Acne') if 'Acne' in class_names else 0
        acne_prob = probabilities[acne_idx].item()

        if acne_prob >= acne_threshold:
            predicted_class = class_names[acne_idx]
            confidence_pct = acne_prob * 100
        else:

            other_idx = 1-acne_idx
            predicted_class = class_names[other_idx]
            confidence_pct = probabilities[other_idx].item() * 100
        
        return predicted_class, confidence_pct 
    

from sklearn.metrics import classification_report, confusion_matrix

all_preds = []
all_labels = []

acne_idx = class_names.index('Acne') if 'Acne' in class_names else 0
ACNE_THRESHOLD = 0.3

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        for i in range(probabilities.size(0)):
            prob_vec = probabilities[i]
            acne_prob = prob_vec[acne_idx].item()
            if acne_prob >= ACNE_THRESHOLD:
                pred_label = acne_idx
            else:
                pred_label = 1 - acne_idx
            all_preds.append(pred_label)
        all_labels.extend(labels.cpu().numpy())

print("\n--- Test Set Evaluation (Threshold = 0.3) ---")
print(confusion_matrix(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=class_names))