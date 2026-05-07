

import numpy as np

import NeuralNetwork as nn
from nn_functions import Activations, ReLU, Min_Max, tanh

test = nn.Model()

test.setLayers([
    Min_Max(),
    ReLU(),
    nn.Layer(3,3),
    ReLU(),
    nn.Layer(3,2),
    ReLU(),
    nn.Layer(2,1)
])

test.evaluateInput([1,2,3])
