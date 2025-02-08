import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from solver import solver


def run_EC_CMS(A, B, k, lambda_):
    A = np.array(A)
    B = np.array(B)

    N = B.shape[0]
    n = 1
    results = np.zeros((N, n))

    # 去除 A 的对角元素
    np.fill_diagonal(A, 0)

    # 调用求解器
    C, _, _ = solver(A, B, lambda_)
    C = np.around(C, decimals=8)
    # N = B.shape[0]
    # n = len(k)
    # results = np.zeros((N, n))
    # A = A - np.diag(np.diag(A))
    #
    # C, _, _ = solver(A, B, lambda_val)

    # for i in range(n):
    #     K = k
    #
    #     C_no_diag = C - np.diag(np.diag(C))
    #     s = squareform(C_no_diag, checks=False)
    #     # s = squareform(C - np.diag(np.diag(C)))
    #
    #     d = 1 - s
    #     Z = np.abs((linkage(d, 'average')))
    #     results[:, i] = fcluster(Z, K, criterion='maxclust')
    #
    #     print('Obtain {} clusters.'.format(K))
    # return results,C

    return C
