# -*- coding: utf-8 -*-
"""Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WKvcMb8oxJcRemD5gOZm0dygjbzelJg3
"""

# clone the repo
!git clone https://github.com/maelfabien/Multimodal-Emotion-Recognition.git

# Commented out IPython magic to ensure Python compatibility.
# %cd Multimodal-Emotion-Recognition

from google.colab import files

uploaded = files.upload()  # Upload your_model.pth from your computer

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install -q kaggle

# Example: Download FER2013 dataset
!kaggle datasets download -d msambare/fer2013

# Unzip
!unzip fer2013.zip -d fer2013

!kaggle datasets download -d davilsena/ckdataset
!unzip ckdataset.zip -d CK+

import pandas as pd
import numpy as np
from PIL import Image
import os

df = pd.read_csv('/content/CK+/ckextended.csv')

emotion_map = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral"  # no contempt
}

output_path = "/content/CK+_filtered_images"
os.makedirs(output_path, exist_ok=True)

for label in emotion_map.values():
    os.makedirs(os.path.join(output_path, label), exist_ok=True)

for idx, row in df.iterrows():
    if row['emotion'] not in emotion_map:
        continue
    emotion = emotion_map[row['emotion']]
    pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(pixels)
    img.save(os.path.join(output_path, emotion, f"{idx}.png"))

import shutil

combined_path = '/content/Combined_Dataset'
shutil.rmtree(combined_path, ignore_errors=True)
os.makedirs(combined_path, exist_ok=True)

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

for emotion in emotions:
    os.makedirs(os.path.join(combined_path, emotion), exist_ok=True)

def copy_images(src, dst):
    for emotion in os.listdir(src):
        if emotion.lower() in emotions:
            src_folder = os.path.join(src, emotion)
            dst_folder = os.path.join(dst, emotion.lower())
            for img in os.listdir(src_folder):
                shutil.copy(os.path.join(src_folder, img), os.path.join(dst_folder, img))

copy_images('/content/fer2013/train', combined_path)
copy_images('/content/fer2013/test', combined_path)
copy_images('/content/CK+_filtered_images', combined_path)

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7  # angry, disgust, fear, happy, neutral, sad, surprise

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Required for pretrained models
    transforms.ToTensor(),
])

# Load dataset
dataset_path = "/content/Combined_Dataset"
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7  # angry, disgust, fear, happy, neutral, sad, surprise

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Required for pretrained models
    transforms.ToTensor(),
])

# Load dataset
dataset_path = "/content/Combined_Dataset"
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split dataset
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

def train_model(model, name, epochs=5):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"🚀 Training {name}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"{name}_emotion_model.pth")
    print(f"✅ Saved {name}_emotion_model.pth\n")

# ResNet18
resnet = models.resnet18(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# VGG16
vgg = models.vgg16(pretrained=True)
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, num_classes)

# MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, num_classes)

# DenseNet121
densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)

# EfficientNet (optional bonus)
efficientnet = models.efficientnet_b0(pretrained=True)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

train_model(EmotionCNN(), "CNN")
train_model(resnet, "ResNet18")
train_model(vgg, "VGG16")
train_model(mobilenet, "MobileNetV2")
train_model(densenet, "DenseNet121")
train_model(efficientnet, "EfficientNetB0")