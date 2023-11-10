from . import Base
import numpy as np


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.softmax_prob = None

    def forward(self, input_tensor):
        # Shifting the input tensor by its maximum value to improve numerical stability
        shifted_input = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        # Computing softmax probabilities
        exp = np.exp(shifted_input)
        self.softmax_prob = exp / np.sum(exp, axis=1, keepdims=True)
        return self.softmax_prob

    def backward(self, error_tensor):
        return self.softmax_prob * (error_tensor - np.sum(error_tensor * self.softmax_prob, axis=1, keepdims=True))
