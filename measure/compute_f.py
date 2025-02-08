def compute_f(T, H):
    if len(T) != len(H):
        print("Size mismatch between T and H")
        print("Size of T:", len(T))
        print("Size of H:", len(H))

    N = len(T)
    numT = 0
    numH = 0
    numI = 0

    for n in range(N):
        Tn = (T[(n+1):] == T[n]).astype(int)
        Hn = (H[(n+1):] == H[n]).astype(int)
        numT += sum(Tn)
        numH += sum(Hn)
        numI += sum(Tn * Hn)

    p = 1
    r = 1
    f = 1

    if numH > 0:
        p = numI / numH

    if numT > 0:
        r = numI / numT

    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)

    return f
