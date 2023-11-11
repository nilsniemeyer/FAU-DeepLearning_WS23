import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_neurons):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_neurons)
        self.out = nn.Linear(hidden_neurons, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


x_axis = np.linspace(-20, 20, 1000)
y_axis = np.sin(x_axis) * np.exp(-x_axis / 10) + (x_axis / 20)**2

inputs_train, inputs_test, labels_train, labels_test = train_test_split(x_axis, y_axis, test_size=0.5, random_state=42)

x_train = torch.FloatTensor(inputs_train).unsqueeze(1)
y_train = torch.FloatTensor(labels_train).unsqueeze(1)
x_test = torch.FloatTensor(inputs_test).unsqueeze(1)
y_test = torch.FloatTensor(labels_test).unsqueeze(1)

predictions = {}
torch.manual_seed(42)

hidden_neurons_list = [1, 5, 10, 50, 100, 250, 500, 1000]

for hidden_neurons in hidden_neurons_list:
    model = Model(1, 1, hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 8000
    for i in range(epochs):
        model.train()
        y_hat = model(x_train).squeeze()
        loss = criterion(y_hat, y_train.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_eval = model(x_test).squeeze().numpy()
        predictions[hidden_neurons] = y_eval
        loss = criterion(torch.FloatTensor(y_eval), y_test.squeeze())
        print(f'With {hidden_neurons} hidden neurons the loss is: {loss.item()}')

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, hidden_neurons in enumerate(hidden_neurons_list):
    ax = axes[i // 4, i % 4]
    ax.scatter(x_test, y_test, label='True Values')
    ax.scatter(x_test, predictions[hidden_neurons], label='Predictions')
    ax.set_title(f'{hidden_neurons} Hidden Neurons', fontsize=20)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, borderaxespad=1.0, fontsize=20)
fig.suptitle('Function Approximation with one hidden layer', fontsize=25, fontweight='bold')
plt.show()
