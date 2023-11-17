from . import Base
import numpy as np


class Dropout(Base.BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.drop_mask = None

    def forward(self, input_tensor):
        # During the training phase, set activations to zero with probability 1-p
        if self.testing_phase:
            # Returning the input tensor without modifying it
            return input_tensor
        else:
            # Create a drop mask that is multiplied with the input tensor to drop activations
            # For the drop mask, generate random values between 0 and 1. If they are below the probability, we keep them and invert the dropout, else we drop them (0 with 1-p)
            self.drop_mask = np.where(np.random.rand(*input_tensor.shape) < self.probability, 1/self.probability, 0)
            return input_tensor * self.drop_mask

    def backward(self, error_tensor):
        return error_tensor * self.drop_mask
