
import numpy as np

class Normalizations:
    def __init__(self):
        return
    def normalize(self, x):
        raise NotImplemented
    def __call__(self, x):
        return self.normalize(x)

class Min_Max(Normalizations):
    def normalize(self, x):
        x_min = x.min()
        x_max = x.max()
        return (x-x_min)/(x_max-x_min)
    
class Standard(Normalizations):
    def normalize(self, x):
        mean = x.mean()
        std = x.std()
        return (x-mean)/std

class Activations:
    def __init__(self):
        return
    def activate(self,x):
        raise NotImplementedError
    def __call__(self, x):
        return self.activate(x)


class ReLU(Activations):
    def activate(self,x):
        return np.maximum(0,x)
    
class tanh(Activations):
    def activate(self, x):
        return np.tanh(x)

class Layer:
    def __init__(self, inputSize, outputSize = None):
        if outputSize == None:
            self.input, self.output = inputSize, inputSize
        else:
            self.input, self.output = inputSize, outputSize
            
    def getInput(self):
        return self.input
    def getSize(self):
        return len(self.input)
    def getOutput(self):
        return self.output