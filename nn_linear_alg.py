import numpy as np
from numpy import array

def randList(size):
    result = []
    for i in range(size):
        result.append(np.random.rand())
    return result

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