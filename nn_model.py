

import numpy as np
from nn_linear_alg import Vector, randMatrix
import nn_functions as f
from layers import normalizationLayer, activationLayer, Layer

class Model:
    def __init__(self):
        self.sequence = []
        self.inputTrace = []
        self.biases = []
        self.weights = []
        
    def setLayers(self, layers):
        for i in range(0,len(layers)):
            curLayer = layers[i]
            if(curLayer.isActivation()):
                self.sequence.append(["Activation", curLayer])
                pass
            elif(curLayer.isNormalization()):
                self.sequence.append(["Normalization", curLayer])
                pass
            else:

                cols = curLayer.getInput()
                rows = curLayer.getOutput()

                self.sequence.append([
                    "Weights",
                    randMatrix(rows,cols).getValues(),
                    randMatrix(rows,cols)
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
        #for layer in self.sequence:
         #   print(layer[0])
    def getTrace(self):
        return self.inputTrace

    def evaluateInput(self, vector):
        #if len(vector) != self.sequence[0].getSize():
        #    raise ValueError("size mismatch")
        #else:
        result = vector
        for i in range(len(self.sequence)):
            self.inputTrace.append([f"{i}th in sequence",result])
            if self.sequence[i][0] == "Weights":
                result = self.sequence[i][2].vectorMultiply(result)
            elif self.sequence[i][0] == "Biases":
                result = f.vectorAddition(result,self.sequence[i][1])
                
            elif self.sequence[i][0] == "Activation":
                for j in range(0, len(result)):
                    result[j] = self.sequence[i][1].activate(result[j])
                    
            elif self.sequence[i][0] == "Normalization":
                result = self.sequence[i][1].normalize(result)
            else:
                print("what?")
                    #result = self.sequence[i][1]
                #result = self.weights[i][2].vectorMultiply(result)
                #result = vectorAddition(result,self.biases[i][1])
            
        print(result)
        return result


# In[606]:


test = Model()
test.setLayers([
    normalizationLayer("min_max"),
    Layer(3,3),
    activationLayer("ReLU"),
    Layer(3,2),
    activationLayer("ReLU"),
    Layer(2,1),
    activationLayer("ReLU")
])

test.getLayers()


# In[607]:


test.evaluateInput([1,2,3])


# In[ ]:




