import numpy as np
from itertools import combinations
import math

from measure.contingency import contingency
def randIndex(c1, c2):
    """
    Calculate Rand Indices to compare two partitions.

    Parameters:
    - c1, c2: numpy arrays listing the class membership.

    Returns:
    - AR: Adjusted Rand index.
    - RI: Unadjusted Rand index.
    - MI: Mirkin's index.
    - HI: Hubert's index.
    """
    if len(c1) != len(c2) or not isinstance(c1, np.ndarray) or not isinstance(c2, np.ndarray):
        raise ValueError('RandIndex: Requires two numpy arrays of the same length')



    C = contingency(c1, c2)
    n = np.sum(C)
    nis = np.sum(np.sum(C, axis=1) ** 2)
    njs = np.sum(np.sum(C, axis=0) ** 2)

    t1 = len(list(combinations(range(int(n)), 2)))  # 将 n 转换为整数
    t2 = np.sum(C ** 2)
    t3 = 0.5 * (nis + njs)


    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))

    A = t1 + t2 - t3
    D = -t2 + t3

    if t1 == nc:
        AR = 0  # 避免除以零；如果 k=1，则定义 Rand = 0
    else:
        AR = (A - nc) / (t1 - nc)
    RI = A / t1
    MI = D / t1
    HI = (A - D) / t1

    return AR

# # Example usage:
# c1 = np.array([0, 0, 1, 1, 2, 2])
# c2 = np.array([0, 0, 1, 2, 2, 1])
#
# AR, RI, MI, HI = randIndex(c1, c2)
# print(f"Adjusted Rand index: {AR}")
# print(f"Unadjusted Rand index: {RI}")
# print(f"Mirkin's index: {MI}")
# print(f"Hubert's index: {HI}")
