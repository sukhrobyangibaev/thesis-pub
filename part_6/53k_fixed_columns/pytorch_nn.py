import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Read dataset
dataset = pd.read_csv('part_6/53k_fixed_columns/53169x156_samples.csv', delimiter=',')

# Encode labels
label_encoder = LabelEncoder()
dataset['winner'] = label_encoder.fit_transform(dataset['winner'])

# Split features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Step 2: Model Definition
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Training Loop
model = NeuralNet(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 64

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Function to calculate accuracy
def calculate_accuracy(predictions, targets):
    binary_predictions = torch.round(torch.sigmoid(predictions))
    correct = (binary_predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy

# Step 4: Evaluation
with torch.no_grad():
    model.eval()
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs.squeeze(), y_val)
    val_accuracy = calculate_accuracy(val_outputs, y_val)
    print(f'Validation Loss: {val_loss.item():.4f}, Accuracy: {val_accuracy:.4f}')