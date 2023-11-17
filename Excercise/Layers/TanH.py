import numpy as np
from . import Base


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = np.tanh(self.input_tensor)
        return self.output_tensor

    def backward(self, error_tensor):
        return error_tensor * (1 - self.output_tensor ** 2)
