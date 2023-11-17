from . import Base
import numpy as np
from . import Initializers
import sys
from icecream import ic
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=500)


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
        #ic("FC: Forward")
        # Adding bias column
        self.input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        #ic("FC: ", self.input_tensor.shape)
        #ic("FC: ", self.input_tensor.shape)
        #ic("FC: ", self.input_tensor)
        # Computing the output tensor by multiplying the input tensor with the weights
        return np.dot(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        #ic("FC: Backward")
        # Performing backward propagation through the layer
        # Computing the error tensor for the previous layer
        prev_error_tensor = np.dot(error_tensor, self.weights[:-1].T)
        # Computing the gradient wrt weights
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        #ic("FC: ", self.gradient_weights.shape)
        # Updating the weights using the optimizer if available
        if self.optimizer is not None:
            #ic("FC: Optimizing Hidden weights")
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return prev_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, self.input_size, self.output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

