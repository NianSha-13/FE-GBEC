import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

# def getAllSegs(baseCls):
#     N, M = baseCls.shape
#     bcs = baseCls.copy()
#
#     nClsOrig = np.max(bcs, axis=0)
#     C = np.cumsum(nClsOrig)
#     bcs = np.add(bcs, np.concatenate(([0], C[:-1])))
#
#     nCls = np.sum(nClsOrig)  # 计算总聚类数
#
#     rows = bcs.flatten()
#     cols = np.tile(np.arange(N), M)
#     data = np.ones_like(rows)  # 使用与 rows 相同形状的全 1 数组
#     baseClsSegs = csr_matrix((data, (rows, cols)), shape=(nCls, N))
#
#     return bcs, baseClsSegs
# def getAllSegs(baseCls):
#     N, M = baseCls.shape
#     # n:    the number of data points.
#     # M:    the number of base clusterings.
#     # baseCls:     the number of clusters (in all base clusterings).
#
#     bcs = np.copy(baseCls)
#     nClsOrig = np.max(bcs, axis=0)  # 每列的最大值，得到每个基础聚类中的最大聚类数。
#     C = np.cumsum(nClsOrig)  # 计算 nClsOrig 的累积和，得到每个基础聚类最大聚类数的累积值。
#     bcs = bcs + np.hstack(([0], C[:-1]))
#     #bcs = np.add(bcs, np.hstack(([0], C[:-1])))  # 构建新矩阵使每一列的不同聚类标签值连续
#     nCls = nClsOrig[-1] + C[-2]  # 计算总的聚类数
#
#     non_zero_values = np.ones(N * M)
#
#     #col_indices = np.repeat(np.arange(N), M)
#     col_indices = np.repeat(np.arange(N) + 1, M)
#     # row=bcs.ravel()
#
#     rows = bcs.flatten()
#     #a = np.max(rows)
#     len1=len(non_zero_values)
#     len2=len(rows)
#     len3=len(col_indices)
#     baseClsSegs = csr_matrix((non_zero_values, (rows, col_indices)), shape=(nCls, N))#row_indices行索引，col_indices列索引
#
#     return bcs, baseClsSegs

def getAllSegs(base_cls):
    N, M = base_cls.shape
    bcs = base_cls.copy()
    n_cls_orig = np.max(bcs+1, axis=0)
    C = np.cumsum(n_cls_orig)
    bcs = bcs + np.concatenate(([0], C[:-1]))
    n_cls = n_cls_orig[-1] + C[-2]

    # base_cls_segs = csr_matrix((np.ones(N*M), (bcs.flatten(), np.repeat(np.arange(1, N+1), M))), shape=(n_cls, N))
    # baseClsSegs = coo_matrix((np.ones(N * M), (bcs.flatten(), np.repeat(np.arange(N), M))), shape=(n_cls, N))
    bcs_flat = bcs.flatten() # 将 bcs 数组展平，并将索引从 MATLAB 的 1-based 转换为 Python 的 0-based

    row_indices = bcs_flat
    col_indices = np.repeat(np.arange(N), M)
    data = np.ones(len(row_indices))

    baseClsSegs = coo_matrix((data, (row_indices, col_indices)), shape=(n_cls, N))
    baseClsSegs = baseClsSegs.tocsr()

    return bcs, baseClsSegs

