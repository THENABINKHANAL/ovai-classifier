import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.io import read_image
import json
from PIL import Image
import math
import numpy as np
import torch.optim as optim
import timm  # Importing the timm library

# Load data
with open('./dataset/label_counts.json', 'r') as f:
    label_counts = json.load(f)
with open('./dataset/labels.json', 'r') as f:
    label_map = json.load(f)

# Calculate class weights for imbalance handling
total_samples = sum(label_counts.values())
class_weights = {class_name.upper(): total_samples / count for class_name, count in label_counts.items()}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')
        label = label_map[item['label'].upper()]
        if self.transform:
            image = self.transform(image)
        return image, label

def train_and_test_model(train_data, test_data, label_map, num_epochs=10, batch_size=32, learning_rate=0.0001):
    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data loaders
    train_dataset = CustomDataset(train_data, transform=data_transforms)
    test_dataset = CustomDataset(test_data, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ViT model
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(label_map))
    model.to(device)

    model.load_state_dict(torch.load(f'vision_transformer_epoch_15.pth', map_location=device))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([class_weights[class_name] for class_name in sorted(label_counts.keys())]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluation
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'vision_transformer_epoch_{epoch+1}.pth')
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f'Test Accuracy: {(correct / total) * 100:.2f}%')

# Load data
train_data = json.load(open('./dataset/train_labels.json', 'r'))
test_data = json.load(open('./dataset/test_labels.json', 'r'))

# Train and evaluate the model
train_and_test_model(train_data, test_data, label_map, num_epochs=150, batch_size=32, learning_rate=0.0001)
