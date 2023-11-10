import numpy as np


class Constant:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        # Filling weights with constant value
        return np.full(weights_shape, self.constant_value)


class UniformRandom:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        # Filling weights with random values in [0, 1)
        return np.random.uniform(0, 1, weights_shape)


class Xavier:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        # Filling weights with values drawn from zero-mean gaussian with std based on the input and output dimensions
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, weights_shape)


class He:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        # Filling weights with values drawn from zero-mean gaussian with std based only on the input dimension
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, weights_shape)
