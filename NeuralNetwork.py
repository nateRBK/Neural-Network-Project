

import numpy as np
from numpy import array
import nn_linear_alg as l
import nn_loss as loss_fs
import optimizer_classes as op
import matplotlib.pyplot as plt
test = op.SGD()


class Normalizations:
    def __init__(self):
        return
    def normalize(self, x):
        raise NotImplementedError
    def __call__(self, x):
        return self.normalize(x)
    def diff(self):
        raise NotImplementedError

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
        self.X = None
        return
    def activate(self,x):
        raise NotImplementedError
    def __call__(self, x):
        return self.activate(x)
    def diff(self):
        raise NotImplementedError


class ReLU(Activations):
    def activate(self,x):
        self.X = x
        return np.maximum(0,x)
    def diff(self):
        result = np.zeros_like(self.X)
        for i in range(0,len(self.X)):
            if self.X[i] >= 0:
                result[i] = 1
        return result
    
class tanh(Activations):
    def activate(self, x):
        self.X = x
        return np.tanh(x)
    def diff(self):
        return 1 - (np.tanh(self.X))**2

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
        self.optimizer = None
        self.biases = []
        self.weights = []
        self.weightLoc = []
        self.biasLoc = []

        self.As = []

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
                matrix = l.randMatrix(rows,cols)
                bias_vector = l.randList(rows)

                self.sequence.append([
                    "Weights",
                    matrix.getValues(),
                    matrix
                ])
                self.weights.append(matrix)

                self.sequence.append([
                    "Biases",
                    bias_vector,
                ])
                self.biases.append(bias_vector)


    def evaluateInput(self, vector):
        result = array(vector)
        self.As= [result]
        a = None

        for i in range(len(self.sequence)):
            self.inputTrace.append([f"{i}th in sequence",result])
            if self.sequence[i][0] == "Weights":
                result = self.sequence[i][2].vectorMultiply(result)
            elif self.sequence[i][0] == "Biases":
                result += self.sequence[i][1]
            elif self.sequence[i][0] == "Activation":
                result = self.sequence[i][1](result)
                a = result.copy()
                self.As.append(a)
                a = None

            elif self.sequence[i][0] == "Normalization":
                result = self.sequence[i][1](result) 
        self.compile()
        #print(result)
        return result

    def compile(self):
        for obj in self.sequence:
            if obj[0] == "Weights" or obj[0] == "Biases":
                obj[1] = np.array(obj[1])
    
    def setOptimizer(self, op):
        self.optimizer = op
    
    def getAs(self):
        return self.As
    
    def update(self, x, y_exp, loss_fn: loss_fs.Loss, lr):
        y_pred = self.evaluateInput(x)
        loss = loss_fn(y_exp, y_pred)
        diff_L = loss_fn.backward()
        cur_err = None
        Sz = None
        curW = None
        k = len(self.As)-2
        errs = []

        for l in range(0, len(self.sequence)):
            j = -1 - l
            if self.sequence[j][0] == "Activation":
                Sz = self.sequence[j][1].diff()
                #print("Sz")
                #print(Sz)
                if l == 0:
                    cur_err = diff_L * Sz
                else:
                    w_t = curW.T
                    cur_err = (w_t@cur_err) * Sz
            elif self.sequence[j][0] == "Biases":
                self.sequence[j][1] = self.sequence[j][1] - lr * cur_err
            elif self.sequence[j][0] == "Weights":
                curW = self.sequence[j][1].copy()
                self.sequence[j][1] = self.sequence[j][1] - lr * np.outer(cur_err, self.As[k])
                k -= 1
            errs.append(cur_err)
        #print("deltas:")
        #print(errs)
        return loss

    """def update(self, x, y_exp, loss_fn: loss_fs.Loss, lr):
        L = len(self.As)-1
        y_pred = self.evaluateInput(x)
        Zs = self.getZ()[::-1]
        loss = loss_fn(y_exp, y_pred)
        diff_L = loss.backward()
        err_L = diff_L * Zs[0][2].diff()
        prev_err = err_L

        for i in range(0, len(Zs)):
            if i == 0:
                cur_err = err_L
            else:
                w = np.array(Zs[i - 1][0])
                w_T = w.T
                cur_err = (w_T @ prev_err)*Zs[i - 1][2].diff()

            w_err = np.outer(cur_err, self.As[-2-i])
            b_err = cur_err


            Zs[i][0] = Zs[i][0] - lr * w_err
            Zs[i][1] = Zs[i][1] - lr * b_err
        return"""
    
    def train(self, x, y_exp, loss_fn: loss_fs.Loss, epochs, lr):
        loss_updates = []
        epoch_numbers = np.arange(1,epochs+1,1)
        for epoch in range(0,epochs):
            cur_train = self.update(x,y_exp,loss_fn,lr)
            loss_updates.append(cur_train)
        loss_updates = array(loss_updates)
        plt.plot(epoch_numbers,loss_updates)
        plt.show()
        
        
    
    
    def getSequence(self, flag = False):
        if flag == True:
            for obj in self.sequence:
                print(obj)
        return self.sequence
    
    def getLayers(self):
        #print(len(self.sequence))
        for i in range(0,len(self.sequence)):
            #print(i)
            print(f"{i} " + self.sequence[i][0])

    def getZ(self):
        result = []
        cur_z = []
        for layer in self.sequence:
            if layer[0] == "Normalization":
                pass
            elif layer[0] == "Weights" or layer[0] == "Biases":
                cur_z.append(layer[1])
            elif layer[0] == "Activation":
                cur_z.append(layer[1])
                result.append(cur_z)
                cur_z = []
        return result
    
    def getTrace(self):
        return self.inputTrace
    
    def getWeights(self):
        for i in range(len(self.weights)):
            print(f"Weights matrix between layers {i+1} and {i+2}")
            print(self.weights[i].getValues())
        return self.weights
    
    def setWeights(self, l, w):
        self.weights[l] = w
    
    def setBiases(self, l, b):
        self.Biases[l] = b
        
    def getBiases(self):
        for i in range(len(self.biases)):
            print(f"Biases vector between layers {i+1} and {i+2}")
            print(self.biases[i])
        return self.biases




