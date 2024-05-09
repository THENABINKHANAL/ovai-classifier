import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import numpy as np

import json

JSON_FILE_TRAIN = "./dataset/labels.json"

with open(JSON_FILE_TRAIN, 'r') as file:
    concept_to_idx = json.load(file)

class CustomDataset(Dataset):
    def __init__(self, json_path, transform=None, concept_to_idx=None, limit=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        if limit is not None:  # If a limit is provided, slice the data
            self.data = self.data[:limit]
        self.transform = transform
        self.concept_to_idx = concept_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['path']).convert('RGB')
        label = self.concept_to_idx[item['label'].upper()]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of EfficientNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the training and testing datasets
train_dataset = CustomDataset(json_path='./dataset/train_labels.json', transform=transform, concept_to_idx=concept_to_idx)
test_dataset = CustomDataset(json_path='./dataset/test_labels.json', transform=transform, concept_to_idx=concept_to_idx)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Load the pre-trained EfficientNet B7 model
model = models.efficientnet_b7(pretrained=True)

# Modify the classifier to output features before classification layer
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove the last layer

device = torch.device("cuda")
model.to(device)

class PseudoTripletLoss(nn.Module):
    """
    This loss function is a pseudo-triplet loss designed to handle scenarios
    where direct triplet formation is challenging due to data limitations.
    It focuses on pairs within a batch to encourage similar embeddings for the 
    same class and dissimilar embeddings for different classes.
    """
    def __init__(self, margin=0.1):
        super(PseudoTripletLoss, self).__init__()
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.relu = nn.ReLU()

    def forward(self, embeddings, labels):
        loss = 0.0
        n = embeddings.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = self.cos_sim(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
                if labels[i] == labels[j]:  # Positive pair
                    # For positive pairs, we want cos_sim to be 1 (similar), so we minimize (1 - cos_sim)
                    loss += self.relu(1 - cos_sim)
                else:  # Negative pair
                    # For negative pairs, we want cos_sim to be less than margin (dissimilar)
                    loss += self.relu(cos_sim - self.margin)
        return loss / (n * (n - 1))  # Normalize the loss

criterion = PseudoTripletLoss(margin=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.0003)

def train_model(model, criterion, optimizer, train_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            embeddings = model(images)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

            if (i + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, '
                      f'Batch Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()  # Set the model to evaluation mode
    device = next(model.parameters()).device  # Get the device the model is on

    similarities = []
    labels_array = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)  # Get the embeddings for the batch

            # Calculate cosine similarity for each pair in the batch
            cos = nn.CosineSimilarity(dim=1)
            n = len(embeddings)
            for i in range(n):
                for j in range(i+1, n):
                    similarity = cos(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
                    similarities.append(similarity)
                    labels_array.append(int(labels[i] == labels[j]))

    similarities = np.array(similarities)
    labels_array = np.array(labels_array)

    # Calculate accuracy based on the threshold
    predictions = (similarities > threshold).astype(np.int_)
    accuracy = np.mean(predictions == labels_array)

    positive_similarities = similarities[labels_array == 1]
    negative_similarities = similarities[labels_array == 0]

    avg_pos_similarity = np.mean(positive_similarities)
    avg_neg_similarity = np.mean(negative_similarities)

    print(f'Average positive pair similarity: {avg_pos_similarity:.4f}')
    print(f'Average negative pair similarity: {avg_neg_similarity:.4f}')
    print(f'Accuracy (threshold={threshold}): {accuracy:.4f}')

    return avg_pos_similarity, avg_neg_similarity, accuracy
model.load_state_dict(torch.load("model_checkpoint_iteration_10.pth"))
if __name__ == '__main__':
    for i in range(10):
        train_model(model, criterion, optimizer, train_loader, epochs=2)
        evaluate_model(model, test_loader)
        torch.save(model.state_dict(), f'./model_checkpoint_iteration_{i+1}.pth')
        optimizer = optim.Adam(model.parameters(), lr=0.0001-i*0.000002)
