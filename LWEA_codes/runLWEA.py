import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def runLWEA(S, ks):
    N = S.shape[0]
    d = stod2(S)  # convert similarity matrix to distance vector

    # average linkage
    Zal = linkage(d, method='average')
    del d
    resultsLWEA = np.zeros((N, 1))


    K = ks
    print('Obtain', K, 'clusters by LWEA.')
    labels = fcluster(Zal, K, criterion='maxclust')

    # 将浮点数数组转换为整数数组
    resultsLWEA[:, 0] = labels.astype(int)

    return resultsLWEA




def stod2(S):
    N = S.shape[0]
    s = np.zeros(N * (N - 1) // 2)
    nextIdx = 0
    for a in range(N - 1):  # change matrix's format to be input of linkage function
        s[nextIdx:nextIdx+(N-a-1)] = np.ravel(S[a, a+1:])
        nextIdx += N - a - 1
    d = 1 - s  # compute distance (d = 1 - sim)
    return d
