import numpy as np


def compute_nmi(T, H):
    N = len(T)
    classes = np.unique(T)
    clusters = np.unique(H)
    num_class = len(classes)
    num_clust = len(clusters)

    # Compute number of points in each class
    D = np.zeros(num_class)
    for j in range(num_class):
        index_class = (T == classes[j])
        D[j] = np.sum(index_class)

    # Mutual information
    mi = 0
    A = np.zeros((num_clust, num_class))
    avgent = 0
    miarr = np.zeros((num_clust, num_class))
    B = np.zeros(num_clust)
    for i in range(num_clust):
        # Number of points in cluster 'i'
        index_clust = (H == clusters[i])
        B[i] = np.sum(index_clust)
        for j in range(num_class):
            index_class = (T == classes[j])
            # Compute number of points in class 'j' that end up in cluster 'i'
            A[i, j] = np.sum(index_class * index_clust)
            if A[i, j] != 0:
                miarr[i, j] = A[i, j] / N * np.log2(N * A[i, j] / (B[i] * D[j]))
                # Average entropy calculation
                avgent = avgent - (B[i] / N) * (A[i, j] / B[i]) * np.log2(A[i, j] / B[i])
            else:
                miarr[i, j] = 0
            mi = mi + miarr[i, j]

    # Class entropy
    class_ent = 0
    for i in range(num_class):
        class_ent = class_ent + D[i] / N * np.log2(N / D[i])

    # Clustering entropy
    clust_ent = 0
    for i in range(num_clust):
        clust_ent = clust_ent + B[i] / N * np.log2(N / B[i])

    # Normalized mutual information
    nmi = 2 * mi / (clust_ent + class_ent)

    return nmi
