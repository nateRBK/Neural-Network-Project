

import numpy as np

class Loss:
    def forward(self, y_exp, y_pred):
        raise NotImplementedError
    def backward(self):
        raise NotImplementedError
    
    def __call__(self, y_exp, y_pred):
        return self.forward(y_exp, y_pred)

class MSE(Loss):
    def forward(self, y_exp, y_pred):
        self.diff = y_pred - y_exp
        return np.mean((self.diff)**2)
    
    def backward(self):
        return 2 * self.diff / np.size(self.diff)
    
    
class MAE(Loss):
    def forward(self, y_exp, y_pred):
        self.diff = y_exp - y_pred
        return np.mean( np.abs(self.diff))
    def backward(self):
        return -1*np.sign(self.diff)/np.size(self.diff)
    

class MBE(Loss):
    def forward(self, y_exp, y_pred):
        self.diff = y_exp - y_pred
        return np.mean(self.diff)
    def backward(self):
        return -1 * np.ones_like(self.diff)/np.size(self.diff)
    
class Huber(Loss):
    def __init__(self):
        self.delta = 0

    def forward(self, y_exp, y_pred, delta):
        self.delta = delta
        condition = np.abs(y_exp - y_pred) < delta
        self.diff = y_exp - y_pred
        l = np. where(condition, 
                      0.5 * (self.diff)**2, 
                      delta * (np.abs(self.diff) - 0.5 * delta)
                      )
        return np.sum(l)/np.size(y_exp)
    
    def backward(self):
        abs_diff = np.abs(self.diff)
        return -1 * np.where(
            abs_diff <= self.delta,
            self.diff,
            self.delta * np.sign(self.diff)
        ) / np.size(self.diff)
    
class Cross_Entropy(Loss):
    def __init__(self):
        self.y_exp, self.y_pred = 0,0
    def forward(self, y_exp, y_pred):
        self.y_exp, self.y_pred = y_exp, y_pred
        return -1*np.sum(y_exp * np.log(y_pred) + (1-y_exp) * np.log(1 - y_pred))/np.size(y_exp)
    
    def backward(self):
        N = np.size(self.y_pred)
        return (1/N) * (-1*self.y_exp /self.y_pred + (1-self.y_exp)/(1-self.y_pred))