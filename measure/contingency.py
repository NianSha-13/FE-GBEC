import numpy as np

def contingency(mem1, mem2):
    """
    Form contingency matrix for two vectors.

    Parameters:
    - mem1, mem2: numpy arrays representing cluster assignments for entities.

    Returns:
    - cont: Contingency matrix.
    """

    if len(mem1) != len(mem2) or not isinstance(mem1, np.ndarray) or not isinstance(mem2, np.ndarray):
        raise ValueError('contingency: Requires two numpy arrays of the same length.')

    cont = np.zeros((int(np.max(mem1)), np.max(mem2)))

    for i in range(len(mem1)):
        cont[int(mem1[i]-1), mem2[i]-1] += 1

    return cont




