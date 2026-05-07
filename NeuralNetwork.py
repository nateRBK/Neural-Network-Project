#!/usr/bin/env python
# coding: utf-8

# In[587]:


import numpy as np
from numpy import array
import nn_functions as f
import nn_linear_alg as l
from nn_functions import ReLU, Activations, Normalizations


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
                    f.randList(rows),
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




