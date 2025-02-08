import numpy as np


def computeLWCA(baseClsSegs, ECI, M):
    baseClsSegs = baseClsSegs.T
    N = baseClsSegs.shape[0]

    LWCA = (baseClsSegs * np.diag(ECI)) @ baseClsSegs.T / M
    np.fill_diagonal(LWCA, 0)
    LWCA += np.eye(N)

    return LWCA
