

import numpy as np
import nn_functions as f

class activationLayer:
    def __init__(self, name: str):
        self.func = f.activations(name)
        
    def activate(self, num):
        return self.func(num)

    def isActivation(self):
        return True

    def isNormalization(self):
        return False

class normalizationLayer:
    def __init__(self, name: str):
        self.func = f.normalizations(name)

    def normalize(self, num):
        return self.func(num)

    def isNormalization(self):
        return True

    def isActivation(self):
        return False


class Layer:
    def __init__(self, inputSize, outputSize = None):
        if outputSize == None:
            self.input, self.output = inputSize, inputSize
            self.isTail = True
        else:
            self.input, self.output = inputSize, outputSize
            
    def getInput(self):
        return self.input
    def getSize(self):
        return self.input
    def isActivation(self):
        return False
    def isNormalization(self):
        return False
    def getOutput(self):
        return self.output
    def tail(self):
        return self.isTail