import numpy as np


def solver(H, A, lambda_val):
    n = A.shape[0]
    t = 0
    e = 1e-2
    max_iter = 100
    I = np.eye(n)
    C = np.zeros((n, n))
    E = np.copy(C)
    F = np.copy(C)
    r1 = 1
    r2 = 1
    Y1 = np.copy(A)
    Y2 = np.copy(C)
    D = np.sum(H, axis=1)
    phi = np.diag(D) - H
    inv_part = np.linalg.inv(2 * phi + (r1 + r2) * I)

    while t < max_iter:
        t += 1

        # update C
        Ct = np.copy(C)
        P1 = A - E + Y1 / r1
        P2 = F - Y2 / r2
        C = np.dot(inv_part, r1 * P1 + r2 * P2)

        # update E
        Et = np.copy(E)
        E = r1 * (A - C) + Y1
        E = E / (lambda_val + r1)
        E[H > 0] = 0

        # update F
        Ft = np.copy(F)
        F = C + Y2 / r2
        F = np.minimum(np.maximum((F + F.T) / 2, 0), 1)

        # update Y
        Y1t = np.copy(Y1)
        residual1 = A - C - E
        Y1 = Y1t + r1 * residual1

        Y2t = np.copy(Y2)
        residual2 = C - F
        Y2 = Y2t + r2 * residual2

        diffC = np.abs(np.linalg.norm(C - Ct, 'fro') / np.linalg.norm(Ct, 'fro'))
        diffE = np.abs(np.linalg.norm(E - Et, 'fro') / np.linalg.norm(Et, 'fro'))
        diffF = np.abs(np.linalg.norm(F - Ft, 'fro') / np.linalg.norm(Ft, 'fro'))
        diffY1 = np.abs(np.linalg.norm(residual1, 'fro') / np.linalg.norm(Y1t, 'fro'))
        diffY2 = np.abs(np.linalg.norm(residual2, 'fro') / np.linalg.norm(Y2t, 'fro'))

        if max([diffC, diffE, diffF, diffY1, diffY2]) < e:
            break

    return C, E, t
