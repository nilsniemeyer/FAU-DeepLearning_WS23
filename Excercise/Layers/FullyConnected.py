from . import Base
import numpy as np
from . import Initializers


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        # Inheriting from the BaseLayer and setting trainable to True, since we want to optimize the weights
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = None

        # Initializing weights
        self.weights = np.zeros((input_size + 1, output_size))
        weights_initializer = Initializers.UniformRandom()
        bias_initializer = Initializers.UniformRandom()

        self.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        # Adding bias column
        self.input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        # Computing the output tensor by multiplying the input tensor with the weights
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        # Performing backward propagation through the layer
        # Computing the error tensor for the previous layer
        prev_error_tensor = np.dot(error_tensor, self.weights[:-1].T)
        # Computing the gradient wrt weights
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # Updating the weights using the optimizer if available
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        return prev_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # Initializing the weights and bias using the initializers
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, self.input_size, self.output_size)

    @property
    def optimizer(self):
        # Get the optimizer associated with the layer
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        # Set the optimizer for the layer
        self._optimizer = value

    @property
    def gradient_weights(self):
        # Get the gradient weights computed during backward propagation
        return self._gradient_weights

