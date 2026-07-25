import os
import cv2
from time import time
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix


def predict_image(img_path, model, transform, class_names, device):
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    IMG_SIZE = 224

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset_base_path = r'C:\asasd\SkinSync\Data\Wrinkles_PostSplit'

    train_ds = datasets.ImageFolder(os.path.join(dataset_base_path, 'train'), transform=train_transform)
    test_ds  = datasets.ImageFolder(os.path.join(dataset_base_path, 'test'), transform=test_transform)
    val_ds   = datasets.ImageFolder(os.path.join(dataset_base_path, 'val'), transform=test_transform)

    class_names = train_ds.classes
    num_classes = len(class_names)

    print('Classes:', class_names)
    print('Class mapping:', train_ds.class_to_idx)
    print('Number of training samples:', len(train_ds))
    print('Number of validation samples:', len(val_ds))
    print('Number of test samples:', len(test_ds))
    print("Samples per class in training:", Counter(train_ds.targets))

    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    for i in range(5, 19):
        for param in model.features[i].parameters():
            param.requires_grad = True

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes) 
    )
    model = model.to(device)

    targets = np.array(train_ds.targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
    weight = 1. / class_sample_count
    print(f"Calculated class weights: {weight}")

    samples_weight = np.array([weight[t] for t in targets])
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, shuffle=False, num_workers=4, pin_memory=use_pin)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=use_pin)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=use_pin)

    backbone_params = []
    for i in range(5, 19):
        backbone_params.extend(list(model.features[i].parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': 5e-6},
        {'params': model.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=1e-4)

    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_acc = 0.0
    best_model_path = 'best_wrinkles_detection_model.pth'

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