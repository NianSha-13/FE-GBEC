import numpy as np
from scipy.sparse import csr_matrix
def getHC(X, bound):
    X_dense = X.toarray()
    E = X_dense.copy()
    E[X_dense >= bound] = 0.0
    A = X_dense - E
    A_sparse = csr_matrix(A)
    # E = np.copy(X)
    # E[X >= bound] = 0.0
    # A = X - E
    return A


