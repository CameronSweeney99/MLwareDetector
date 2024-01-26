#train_nerual_network.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np  

class MyDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)

        # Manually set the numeric labels based on the label mapping
        label_mapping = {'benign': 0, 'malware': 1}
        data['label'] = data.iloc[:, -1].map(label_mapping)

        # Check if all labels have been converted correctly
        if data['label'].isnull().any():
            raise ValueError("Non-numeric data found in labels column.")

        self.X = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(data['label'].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)  
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, num_classes)  

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    # Load datasets
    train_dataset = MyDataset('../data/training_extracted_features.csv')
    val_dataset = MyDataset('../data/validation_extracted_features.csv')
    test_dataset = MyDataset('../data/test_extracted_features.csv')

    # Create DataLoader instances for each dataset
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # Model parameters
    input_size = train_dataset.X.shape[1]
    num_classes = 2  

    # Initialize the model
    model = NeuralNet(input_size, num_classes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Evaluate the model on the test data
    test_loss, test_accuracy = validate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), '../models/neural_network.pth')
    print("Model saved in '../models/neural_network.pth'")

if __name__ == "__main__":
    main()