from . import Base
import numpy as np

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        # Applying ReLU activation function
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        # Backward propagation of the ReLU layer
        # The ReLU function is piecewise differentiable with 0 for e<=0 and 1 for e>0
        # This means the error_tensor of the following layer can be multiplicated elementwise with either 0 or 1
        return np.multiply(error_tensor, np.where(self.input_tensor > 0, 1, 0))
