import numpy as np


class CrossEntropyLoss():
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # Computing the cross entropy loss
        self.prediction_tensor = prediction_tensor
        # Element-wise multiplication with the label_tensor to only compute the loss from the correct classes
        return np.sum(-np.log(prediction_tensor + self.eps) * label_tensor)

    def backward(self, label_tensor):
        # Computing the first error_tensor
        return - label_tensor / (self.prediction_tensor + self.eps)


