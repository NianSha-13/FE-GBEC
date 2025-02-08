import numpy as np

def getCA(baseClsSegs, M):
    CA = np.dot(baseClsSegs.T, baseClsSegs) / M
    return CA
