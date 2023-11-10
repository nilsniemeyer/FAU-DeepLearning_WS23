import numpy as np


class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        weight_tensor += self.v
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.eps = np.finfo(float).eps
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.iteration = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.iteration += 1
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * gradient_tensor ** 2
        v_hat = self.v / (1 - self.mu ** self.iteration)
        r_hat = self.r / (1 - self.rho ** self.iteration)
        weight_tensor -= self.learning_rate * v_hat / (np.sqrt(r_hat) + self.eps)
        return weight_tensor
