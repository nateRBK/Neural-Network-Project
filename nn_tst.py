

import numpy as np

import NeuralNetwork as nn
#from nn_functions import Activations, ReLU, Min_Max, tanh
from optimizer_classes import SGD
from nn_loss import MSE

test = nn.Model()

test.setLayers([
    nn.Min_Max(),
    nn.Layer(3,2),
    nn.ReLU(),
    nn.Layer(2,1),
    nn.ReLU(),
])
x = [1,2,3]
#test.evaluateInput([1,2,3])
#test.update(x,1,MSE(),.01)
#print("new value")
#test.evaluateInput(x)
test.train(x,1.2,MSE(),1000,0.01)