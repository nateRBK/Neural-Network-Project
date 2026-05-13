

import numpy as np

class Optimization:
    def update(self, weights, grad):
        raise NotImplementedError
    def __call__(self, weights, grad):
        return self.update(weights, grad)

class SGD(Optimization):
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, weights, grad):
        return weights - self.lr*grad
    def test(self):
        print("weh")