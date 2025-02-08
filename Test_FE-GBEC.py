
import time
import pandas as pd
from sklearn.cluster import KMeans
from LWEA_codes.computeECI import computeECI
from LWEA_codes.computeLWCA import computeLWCA
from LWEA_codes.getAllSegs import getAllSegs
from measure.randIndex import randIndex
from measure.compute_f import compute_f
from measure.compute_nmi import compute_nmi

from splitGBs import splitGBs
from getCA import getCA
from getHC import getHC
from run_EC_CMS import run_EC_CMS
from model import GCN, Model
import torch
import warnings


warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import warnings
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from measure.loss import *

warnings.filterwarnings('ignore')

# 忽略所有 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
alpha = 0.8  # Initialization parameter alpha
_lambda = 0.55
# seed = 64
# np.random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    file_path = r"F:\学术\FE-GBEC\FE-GBEC\dataset\3MC.csv"
    data = pd.read_csv(file_path, header=None)
    #data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    members = data.iloc[:, :-1].values
    labels = data.iloc[:, -1]
    labels = np.array(labels)
    gt = labels
    clsNums = len(np.unique(labels))
    all_indices = list(range(len(data)))
    # 设置聚类参数
    num_iterations = 20
    num_clusters = clsNums
    num_iterations = num_iterations  # 我们要运行 k-means 算法的次数
    N, poolSize = members.shape
    start_time = time.time()
    # 粒球分裂
    gb_list, gb_index = splitGBs(members, all_indices)
    print(gb_index)

    # 确保 gb_list 中的粒球数量多于 num_clusters
    assert len(
        gb_list) >= num_clusters, "The number of granular balls must be greater than or equal to the number of clusters."
    # 初始化聚类矩阵，每行一个样本点，每列一次聚类的结果
    cluster_matrix = np.zeros((len(gb_list), num_iterations), dtype=int)

    # 循环 k-means 构造聚类矩阵
    for iteration in range(num_iterations):
        # 随机选择 num_clusters 个粒球中心作为初始质心
        selected_gb_indices = np.random.choice(len(gb_list), size=num_clusters, replace=False)
        initial_centroids = np.array([gb_list[i].center for i in selected_gb_indices])

        # 从gb_list中提取所有球的中心点
        centroids = [gb.center for gb in gb_list]

        # 转换为NumPy数组
        data_array = np.array(centroids)

        # 创建 k-means 实例
        kmeans = KMeans(n_clusters=num_clusters, init=initial_centroids, n_init=1, max_iter=300)

        # 拟合 k-means 模型
        kmeans.fit(data_array)

        # 保存聚类结果到聚类矩阵
        cluster_matrix[:, iteration] = kmeans.labels_
    # 调用EC-CMS方法
    bcs, baseClsSegs = getAllSegs(cluster_matrix)
    para_theta = 0.4
    M = 20
    ECI = computeECI(bcs, baseClsSegs, para_theta)
    LWCA = computeLWCA(baseClsSegs, ECI, M)
    CA = getCA(baseClsSegs, M)
    A = getHC(CA, alpha)
    CA_matrix = run_EC_CMS(A, LWCA, clsNums, _lambda)
    # 生成邻接矩阵，假设相似度大于0.5时认为有边连接
    CA_matrix_tensor = torch.tensor(CA_matrix).float()

    # 将邻接矩阵转换为边索引（edge_index），PyTorch Geometric要求边索引的格式为 [2, E]，其中 E 是边的数量
    edge_index = torch.nonzero(CA_matrix_tensor > 0).t().contiguous()  # 获取非零元素的索引
    edge_index = edge_index.to(torch.long)  # 转换为long类型的边索引
    center_list = [gb.center for gb in gb_list]

    center_list = np.array(center_list)
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(center_list)
    X = torch.Tensor(X)
    # 创建图数据对象
    data = Data(x=X, edge_index=edge_index)
    # 创建GCN模型
    in_channels = len(center_list[0])  # 每个节点的特征维度
    out_channels = len(center_list[0])  # 输出特征维度
    model = Model(in_channels=in_channels, out_channels=out_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # 假设标签（目标输出），这里只是随机生成的标签用于训练
    labels = None

    # 训练循环
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)  # 获取模型输出（节点的嵌入表示）

        # 确保 centerList 是 Tensor
        centerList_tensor = torch.tensor(center_list, dtype=torch.float32, device=out.device)

        # 计算软分配矩阵 Q
        q = 1.0 / (1.0 + torch.sum(torch.pow(out.unsqueeze(1) - centerList_tensor.unsqueeze(0), 2), 2))
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / q.sum(1)).t()

        # 计算目标分布 P
        p = target_distribution(q.detach())

        # 计算 KL 散度作为损失
        kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
        kl_loss.backward(retain_graph=True)  # 保留计算图
        optimizer.step()

        print(f"Epoch {epoch}, KL Loss: {kl_loss.item()}")

    end_time = time.time()
    kmeans = KMeans(n_clusters=clsNums)
    node_embeddings = out.detach().numpy()  # 将节点嵌入转为 numpy 数组
    kmeans.fit(node_embeddings)
    # 获取聚类标签
    cluster_labels = kmeans.labels_

    # 对数据标签进行映射
    n_points = len(members)
    mapped_labels = [None] * n_points  # 创建一个大小为 9 的空列表，用于存储每个数据点的标签
    # 遍历 gb_index 和 gb_list
    for i, indices in enumerate(gb_index):
        # 如果 indices 是整数，将其转换为包含一个元素的列表
        if isinstance(indices, int):
            indices = [indices]

        # 将 gb_list 中粒球 i 的标签赋给 gb_index 中粒球 i 包含的所有数据点
        for index in indices:
            mapped_labels[index] = cluster_labels[i]

    ARI = adjusted_rand_score(gt, mapped_labels)
    NMI = normalized_mutual_info_score(gt, mapped_labels)
    F = compute_f(gt, mapped_labels)

    print('ARI', ARI)
    print('NMI', NMI)
    print('F-score', F)

    execution_time = end_time - start_time
    print(f"程序运行时间：{execution_time}")