import copy
import numpy as np
from . import Base
from .TanH import TanH
from .Sigmoid import Sigmoid
from .FullyConnected import FullyConnected
from icecream import ic

np.set_printoptions(precision=2)


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self._memorize = False
        self._optimizer = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the hidden state with all zeros
        self.input_tensor = None
        self.T = None

        self.h_t = None
        self.y_t = None

        self.prev_h_t = None

        self.FC_hidden_memory = None
        self.FC_hidden = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.weights = self.FC_hidden.weights
        self.hidden_gradient_weights = None

        self.tanh = TanH()

        self.FC_output_memory = None
        self.FC_output = FullyConnected(self.hidden_size, self.output_size)
        self.output_gradient_weights = None

        self.sigmoid = Sigmoid()

    def forward(self, input_tensor):
        #ic("RNN: Forward")
        self.input_tensor = input_tensor
        self.T = self.input_tensor.shape[0]
        #ic("RNN: T", self.T)
        self.y_t = np.zeros((self.T, self.output_size))
        self.h_t = np.zeros((self.T + 1, self.hidden_size))

        if self.memorize and self.prev_h_t is not None:
            self.h_t[0] = self.prev_h_t

        self.FC_hidden_memory = []
        self.FC_output_memory = []

        for t in range(self.T):
            concatenated_input = np.concatenate((self.input_tensor[t], self.h_t[t])).reshape(1, -1)

            u_t = self.FC_hidden.forward(concatenated_input)
            self.FC_hidden_memory.append(self.FC_hidden.input_tensor)

            self.h_t[t+1] = self.tanh.forward(u_t)

            o_t = self.FC_output.forward(self.h_t[t+1].reshape(1, -1))
            self.FC_output_memory.append(self.FC_output.input_tensor)

            self.y_t[t] = self.sigmoid.forward(o_t)

        self.prev_h_t = self.h_t[-1]
        return self.y_t

    def backward(self, error_tensor):
        #ic("RNN: Backward")
        previous_hidden_state_error = np.zeros((1, self.hidden_size))
        output_error_tensor = np.zeros((self.T, self.input_size))

        self.hidden_gradient_weights = np.zeros_like(self.FC_hidden.weights)
        self.output_gradient_weights = np.zeros_like(self.FC_output.weights)

        for t in reversed(range(self.T)):
            self.sigmoid.output_tensor = self.y_t[t]
            derivative_sigmoid = self.sigmoid.backward(error_tensor[t].reshape(1, -1))

            self.FC_output.input_tensor = self.FC_output_memory[t]
            derivative_wrt_o_t = self.FC_output.backward(derivative_sigmoid)
            self.output_gradient_weights += self.FC_output.gradient_weights

            error_before_tanh = previous_hidden_state_error + derivative_wrt_o_t

            self.tanh.output_tensor = self.h_t[t + 1]
            error_after_tanh = self.tanh.backward(error_before_tanh)

            self.FC_hidden.input_tensor = self.FC_hidden_memory[t]
            combined_hidden_error = self.FC_hidden.backward(error_after_tanh)
            self.hidden_gradient_weights += self.FC_hidden.gradient_weights

            input_error = combined_hidden_error[:, :self.input_size]
            previous_hidden_state_error = combined_hidden_error[:, self.input_size:]

            output_error_tensor[t] = input_error

        if self.optimizer is not None:
            ic("RNN: Optimizing Hidden weights")
            self.FC_hidden.weights = self._optimizer.calculate_update(self.FC_hidden.weights, self.hidden_gradient_weights)
            self.FC_output.weights = self._optimizer.calculate_update(self.FC_output.weights, self.output_gradient_weights)
        return output_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.FC_hidden.initialize(weights_initializer, bias_initializer)
        self.FC_output.initialize(weights_initializer, bias_initializer)
        self.weights = self.FC_hidden.weights

    def calculate_regularization_loss(self):
        return self.optimizer.regularizer.norm(self.FC_hidden.weights)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def weights(self):
        return self.FC_hidden.weights

    @weights.setter
    def weights(self, weights):
        self.FC_hidden.weights = weights

    @property
    def gradient_weights(self):
        return self.hidden_gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_hidden._gradient_weights = gradient_weights
