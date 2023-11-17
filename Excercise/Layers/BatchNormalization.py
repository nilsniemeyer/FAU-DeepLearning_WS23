import copy

from . import Base, Helpers
import numpy as np
from icecream import ic


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.input_tensor = None
        self.input_save_for_backward = None
        self.output_tensor = None
        self.weights = None
        self.bias = None
        self.batch_mean = None
        self.batch_moving_mean = None
        self.batch_var = None
        self.batch_moving_var = None
        self.eps = np.finfo(float).eps

        self.initialize(None, None)

        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)
        self._optimizer = None
        self._bias_optimizer = None

    def forward(self, input_tensor):
        if input_tensor.ndim == 4:
            self.input_tensor = self.reformat(input_tensor)
        else:
            self.input_tensor = input_tensor

        if self.batch_moving_mean is None:
            self.batch_moving_mean = np.mean(self.input_tensor, axis=0)
            self.batch_moving_var = np.var(self.input_tensor, axis=0)

        if self.testing_phase:
            self.batch_mean = self.batch_moving_mean
            self.batch_var = self.batch_moving_var
        else:
            self.batch_mean = np.mean(self.input_tensor, axis=0)
            self.batch_var = np.var(self.input_tensor, axis=0)
            self.batch_moving_mean = 0.8 * self.batch_moving_mean + self.batch_mean * (1 - 0.8)
            self.batch_moving_var = 0.8 * self.batch_moving_var + self.batch_var * (1 - 0.8)

        self.output_tensor = (self.input_tensor - self.batch_mean) / np.sqrt(self.batch_var + self.eps)

        if input_tensor.ndim == 4:
            self.output_tensor = self.reformat(self.output_tensor)
            return self.weights.reshape((1, self.channels, 1, 1)) * self.output_tensor + self.bias.reshape((1, self.channels, 1, 1))
        else:
            return self.weights * self.output_tensor + self.bias

    def backward(self, error_tensor):
        if error_tensor.ndim == 4:
            gradient_wrt_inputs = Helpers.compute_bn_gradients(self.reformat(error_tensor), self.input_tensor, self.weights, self.batch_mean, self.batch_var)
            gradient_wrt_inputs = self.reformat(gradient_wrt_inputs)
            self.gradient_weights = np.sum(error_tensor * self.output_tensor, axis=(0, 2, 3))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            gradient_wrt_inputs = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.batch_mean, self.batch_var)
            self.gradient_weights = np.sum(error_tensor * self.output_tensor, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return gradient_wrt_inputs

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        if tensor.ndim == 4:
            # Save original dimensions for the 4D tensor (needed for reconstruction from 2D tensor)
            self.original_dims = tensor.shape
            B, H, M, N = tensor.shape
            # Reshape from B × H × M × N to B × H × M · N
            tensor = tensor.reshape(B, H, M * N)
            # Transpose from B × H × M · N to B × M · N × H
            tensor = tensor.transpose(0, 2, 1)
            # Reshape to B · M · N × H
            tensor = tensor.reshape(B * M * N, H)
        elif tensor.ndim == 2 and self.original_dims is not None:
            # Retrieve original dimensions for 4D tensor
            B, H, M, N = self.original_dims
            # Reshape from B · M · N × H to B × M · N × H
            tensor = tensor.reshape(B, M * N, H)
            # Transpose from B × M · N × H to B × H × M · N
            tensor = tensor.transpose(0, 2, 1)
            # Reshape back to B × H × M × N
            tensor = tensor.reshape(B, H, M, N)
        else:
            raise ValueError("Unsupported tensor shape or missing original dimensions.")

        return tensor


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer
    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value