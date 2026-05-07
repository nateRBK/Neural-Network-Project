import numpy as np
from numpy import array

class Vector:
    def __init__(self, data):
        self.values = []
        if isinstance(data, int):
            for i in range(0,data):
                self.values.append(np.random.rand()*2 - 1)
        elif isinstance(data, list):
            self.values = data
        else:
            raise TypeError("weh")
    def getValues(self):
        return np.array(self.values)
    def getLength(self):
        return len(self.values)

class randMatrix:
    def __init__(self, m, n):
        self.val = []
        self.rows = m
        self.cols = n
        for i in range(0,m):
            temp = []
            for j in range(0, n):
                temp.append(np.random.rand()*2 - 1)
            self.val.append(temp)
    def getValues(self):
        return self.val

    def vectorMultiply(self, vec):
        if len(vec) != self.cols:
            raise ValueError("Vector size mismatch")
        else:
            result = []
            for row in self.val:
                temp = 0
                for i in range(len(row)):
                    temp += row[i] * vec[i]
                result.append(temp)
            return np.array(result)