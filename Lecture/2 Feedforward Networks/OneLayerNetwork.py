import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from icecream import ic

class OneLayerNetwork:
    def __init__(self, inputs, labels, hidden_layer_neurons, learning_rate, num_epochs):
        input_train, input_test, label_train, label_test = train_test_split(inputs, labels, test_size=0.3,
                                                                            random_state=42)
        self.input_train = np.hstack((input_train, np.ones((input_train.shape[0], 1))))
        self.label_train = label_train
        self.input_test = np.hstack((input_test, np.ones((input_test.shape[0], 1))))
        self.label_test = label_test
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        weights_bias_state = np.random.RandomState()
        self.hidden_weights = weights_bias_state.rand(self.input_train.shape[1], hidden_layer_neurons)
        self.output_weights = weights_bias_state.rand(hidden_layer_neurons + 1, 2)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        hidden_output = np.dot(self.input_tensor, self.hidden_weights)
        self.ReLU_output = np.append(self.ReLU(hidden_output), 1)
        self.output = np.dot(self.ReLU_output, self.output_weights)
        prediction = self.SoftMax(self.output)
        return prediction

    def backward(self, prediction_tensor, label_tensor):
        loss_wrt_sf_pred = - label_tensor / (prediction_tensor + np.finfo(float).eps) # Derivative Loss wrt SoftMax predictions
        sf_pred_wrt_outputs = prediction_tensor * (loss_wrt_sf_pred - np.sum(loss_wrt_sf_pred * prediction_tensor)) # Derivative SoftMax wrt outputs
        outputs_wrt_ReLU = np.dot(sf_pred_wrt_outputs, self.output_weights[:-1].T) # Derivative outputs wrt ReLU outputs
        gradient_output_weights = np.dot(self.ReLU_output.reshape(-1, 1), sf_pred_wrt_outputs.reshape(1, -1)) # Derivative outputs wrt output weights
        self.output_weights = self.output_weights - self.learning_rate * gradient_output_weights # Update output weights
        ReLU_wrt_hidden_output = np.multiply(outputs_wrt_ReLU, np.where(self.ReLU_output[:-1] > 0, 1, 0)) # Derivative ReLU wrt hidden layer
        gradient_hidden_weights = np.dot(self.input_tensor.reshape(-1, 1), ReLU_wrt_hidden_output.reshape(1, -1)) # Derivative Hidden layer wrt hidden weights
        self.hidden_weights = self.hidden_weights - self.learning_rate * gradient_hidden_weights  # Update hidden weights

    def ReLU(self, input_tensor):
        return np.maximum(0, input_tensor)

    def SoftMax(self, input_tensor):
        # Shifting the input tensor by its maximum value to improve numerical stability
        shifted_input = input_tensor - np.max(input_tensor)
        # Computing softmax probabilities
        exp = np.exp(shifted_input)
        softmax_prob = exp / np.sum(exp)
        return softmax_prob

    def loss(self, prediction_tensor, label_tensor):
        return np.sum(-np.log(prediction_tensor + np.finfo(float).eps) * label_tensor)

    def train(self):
        loss_progress = []
        num_train_samples = self.input_train.shape[0]

        for epoch in range(self.num_epochs):

            perm = np.random.permutation(num_train_samples)
            self.input_train = self.input_train[perm]
            self.label_train = self.label_train[perm]

            for iteration in range(num_train_samples):
                i_input = self.input_train[iteration]
                # Basically one-hot-encoding the label
                if self.label_train[iteration] == 0:
                    label_tensor = np.array([1.0, 0.0])
                else:
                    label_tensor = np.array([0.0, 1.0])

                prediction_tensor = self.forward(i_input)
                loss = self.loss(prediction_tensor, label_tensor)
                self.backward(prediction_tensor, label_tensor)
                loss_progress = np.append(loss_progress, loss)

        self.test()
        self.visualize(loss_progress)

    def test(self):
        num_test_samples = self.input_test.shape[0]
        correct_predictions = 0

        for iteration in range(num_test_samples):
            i_input = self.input_test[iteration]
            if self.label_test[iteration] == 0:
                label_tensor = np.array([1.0, 0.0])
            else:
                label_tensor = np.array([0.0, 1.0])
            prediction_tensor = np.round(self.forward(i_input))
            if np.array_equal(prediction_tensor, label_tensor):
                correct_predictions += 1
        print(correct_predictions)

        self.accuracy = (correct_predictions / num_test_samples) * 100




    def visualize(self, loss_progress):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_progress, color='blue', label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Progress, with test accuracy: {}'.format(self.accuracy))
        plt.legend()
        plt.grid(True)
        plt.show()


data = np.genfromtxt("data_banknote_authentication.csv", delimiter=',')
np.random.shuffle(data)
# Each sample has four input variables
input_data = data[:, 0:4]
label_data = data[:, 4]

network = OneLayerNetwork(input_data, label_data, 2, 0.01, 3)
network.train()
