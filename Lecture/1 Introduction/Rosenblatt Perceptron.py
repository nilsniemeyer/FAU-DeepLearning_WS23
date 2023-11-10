import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class RosenblattPerceptron:
    # The Rosenblatt perceptron allows for a simple binary classification (that is not good for training)
    def __init__(self, inputs, labels):
        # It takes an input feature vector x to which we need to add a bias of 1
        inputs = np.insert(inputs, 0, 1, axis=1)
        # And it takes a weight vector w. We start with random weights
        self.w = np.random.uniform(-1, 1, size=inputs.shape[1])
        # And split training and test data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(inputs, labels, test_size=0.3)
        # Create prediction vectors y_hat to evaluate the performance
        # They are assigned random classes to ignore performance increases based on simply filling the array
        self.y_hat_train = np.random.choice([-1, 1], size=self.y_train.shape)
        self.y_hat_test = np.random.choice([-1, 1], size=self.y_test.shape)
        self.training_history = []
        self.test_history = []

    def activation(self, input_it):
        # It multiplies each value from x with a weight from w and sums them up (vector notation: w^t*x)
        activation_input = np.dot(self.w.T, input_it)
        # It uses a sign function as an activation function
        # If the activation_input is positive, the activation_output will be 1, if negative then -1
        class_prediction = np.sign(activation_input)
        return class_prediction

    def objective_function(self, label, prediction):
        # The perceptron objective function is to maximize the accuracy or minimize the number of wrong classifications:
        # -1 * SUM(label * prediction) for all misclassified samples.
        # Let's first filter the misclassified samples (this is very unnecessary, but still)
        misclassified_indices = np.where(label * prediction <= 0)[0]
        misclassified_label = label[misclassified_indices]
        misclassified_prediction = prediction[misclassified_indices]
        # And then (basically) calculate the number of wrong classifications
        wrong_classifications = -1 * np.sum(misclassified_label * misclassified_prediction)
        return wrong_classifications

    def training(self, epochs):
        # Training Strategy 2: Take an update step right after each misclassified sample
        for e in np.arange(epochs):
            # Shuffle the order in which the samples are processed.
            indices = np.arange(self.x_train.shape[0])
            np.random.shuffle(indices)
            # For each sample, get the input, label and prediction
            for k in indices:
                input_it = self.x_train[k, :]
                label_it = self.y_train[k]
                pred_it = self.activation(input_it)
                # Compare the label and prediction and update the weights
                if label_it != pred_it:
                    # If the prediction is wrong, the weights are shifted into the direction of the correct
                    # classification with the input magnitude. This puts emphasis on the features which were most
                    # important for the wrongly classified sample.
                    self.w = self.w + label_it * input_it
                # Save the prediction and evaluate the performance (as is: evaluate test performance after each sample)
                self.y_hat_train[k] = pred_it
                self.training_history.append(self.objective_function(self.y_train, self.y_hat_train))
                self.test()
                self.test_history.append(self.objective_function(self.y_test, self.y_hat_test))

    def test(self):
        # Full classification of the test set
        for k in range(0, self.x_test.shape[0]):
            input_it = self.x_test[k, :]
            pred_it = self.activation(input_it)
            self.y_hat_test[k] = pred_it


data = np.genfromtxt("data_banknote_authentication.csv", delimiter=',')
input_data = data[:, 0:4]
label_data = data[:, 4]

perceptron = RosenblattPerceptron(input_data, label_data)
perceptron.training(2)

plt.figure(figsize=(10, 5))
plt.plot(perceptron.training_history, label='Training History')
plt.plot(perceptron.test_history, label='Test History')
plt.xlabel('Iteration')
plt.legend()
plt.show()

