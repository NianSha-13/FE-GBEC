import numpy as np


def computeECI(bcs, baseClsSegs, para_theta):
    M = bcs.shape[1]  # Number of base clusterings
    ETs = getAllClsEntropy(bcs, baseClsSegs)  # Get the entropy of each cluster
    ECI = np.exp(-ETs / para_theta / M)  # Compute ECI
    return ECI


def getAllClsEntropy(bcs, baseClsSegs):
    baseClsSegs = baseClsSegs.T  # Transpose baseClsSegs
    nCls = baseClsSegs.shape[1]  # Get the number of clusters
    Es = np.zeros(nCls)  # Initialize array to store cluster entropies

    for i in range(nCls):
        partBcs = bcs[np.nonzero(baseClsSegs[:, i] != 0)[0], :]  # Get base clusterings for the current cluster
        Es[i] = getOneClsEntropy(partBcs)  # Compute entropy for the current cluster

    return Es


def getOneClsEntropy(partBcs):
    E = 0  # Initialize cluster entropy

    for i in range(partBcs.shape[1]):
        tmp = np.sort(partBcs[:, i])  # Sort cluster elements
        uTmp = np.unique(tmp)  # Get unique elements

        if len(uTmp) <= 1:  # Skip if only one unique element
            continue

        cnts = np.zeros_like(uTmp)  # Initialize array to store counts

        for j in range(len(uTmp)):
            cnts[j] = np.sum(np.sum(tmp == uTmp[j]))  # Count occurrences of each unique element

        cnts = cnts / np.sum(cnts)  # Normalize counts
        E = E - np.sum(cnts * np.log2(cnts))  # Compute entropy using Shannon entropy formula

    return E
