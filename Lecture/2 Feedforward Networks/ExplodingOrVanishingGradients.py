import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_neurons):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc4 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc5 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc6 = nn.Linear(hidden_neurons, hidden_neurons)
        self.out = nn.Linear(hidden_neurons, output_size)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = F.tanh(self.fc6(x))
        y = self.out(x)
        return y


# Load and preprocess the data
wine = load_wine()
X = wine.data
Y = wine.target

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
Y_train = torch.LongTensor(Y)

# Convert the data to a PyTorch Dataset
train_dataset = TensorDataset(X_train, Y_train)

# Create a DataLoader with a batch size of 1
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

# Set the random seed for reproducibility
torch.manual_seed(43)

# Define the list of learning rates to iterate over
learning_rate_list = [0.000001, 0.0001, 0.01, 0.1, 1, 10]

# Create a figure with subplots
fig, axes = plt.subplots(len(learning_rate_list), 1, figsize=(10, 20))

# Train the model and record the gradients
for i, learning_rate in enumerate(learning_rate_list):
    model = Model(13, 3, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    gradients_sums_per_layer = {f'fc{j}': [] for j in range(1, 7)}

    epochs = 1
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Record the sum of gradients for each layer
            for j in range(1, 7):
                layer = getattr(model, f'fc{j}')
                gradients_sums_per_layer[f'fc{j}'].append(layer.weight.grad.abs().sum().item())

        # Plot the sum of gradients for each layer
        for j in range(1, 7):
            axes[i].plot(gradients_sums_per_layer[f'fc{j}'], label=f'Layer {j}')

    axes[i].set_title(f'Learning Rate = {learning_rate}')
    axes[i].set_xlabel('Epochs')
    axes[i].set_ylabel('Sum of Gradients')
    axes[i].legend()

plt.tight_layout()
plt.show()
