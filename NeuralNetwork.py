#!/usr/bin/env python
# coding: utf-8

# In[587]:


import numpy as np
from numpy import array
import nn_linear_alg as l

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

class Model:
    def __init__(self):
        self.sequence = []
        self.inputTrace = []
        
    def setLayers(self, layers):
        for i in range(0,len(layers)):
            curLayer = layers[i]
            if(isinstance(curLayer, Activations)):
                self.sequence.append(["Activation", curLayer])
            elif(isinstance(curLayer, Normalizations)):
                self.sequence.append(["Normalization", curLayer])
            else:

                cols = curLayer.getInput()
                rows = curLayer.getOutput()

                self.sequence.append([
                    "Weights",
                    l.randMatrix(rows,cols).getValues(),
                    l.randMatrix(rows,cols)
                ])

                self.sequence.append([
                    "Biases",
                    l.randList(rows),
                ])
            
                
                
    def getSequence(self):
        return self.sequence
    
    def getLayers(self):
        for i in range(0,len(self.sequence)):
            print(f"{i} " + self.sequence[i][0])

    def getTrace(self):
        return self.inputTrace

    def evaluateInput(self, vector):

        result = array(vector)
        for i in range(len(self.sequence)):
            self.inputTrace.append([f"{i}th in sequence",result])
            if self.sequence[i][0] == "Weights":
                result = self.sequence[i][2].vectorMultiply(result)

            elif self.sequence[i][0] == "Biases":
                result += self.sequence[i][1]

            elif self.sequence[i][0] == "Activation":
                result = self.sequence[i][1](result)
                #for j in range(0, len(result)):
                #    result[j] = self.sequence[i][1].activate(result[j])
                    
            elif self.sequence[i][0] == "Normalization":
                result = self.sequence[i][1](result)
            else:
                print("what?")          
        print(result)
        return result




