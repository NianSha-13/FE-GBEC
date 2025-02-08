import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


def contrastive_loss(embeddings, temperature=0.5):
    """
    自监督对比损失：最大化同类节点嵌入的相似性，最小化不同类节点的相似性。

    embeddings: 图节点的嵌入表示，shape = (num_nodes, embedding_dim)
    temperature: 控制相似性力度的温度系数
    """
    # 计算相似度矩阵（这里使用的是余弦相似度）
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

    # 构造一个掩码，值为1表示节点属于同一类别，值为0表示属于不同类别
    mask = torch.eye(sim_matrix.size(0)).float().to(sim_matrix.device)  # 对角线是同类节点

    # 最大化正样本相似度，最小化负样本相似度
    positive_sim = sim_matrix * mask  # 同类节点的相似度
    negative_sim = sim_matrix * (1 - mask)  # 不同类节点的相似度

    # 计算对比损失
    # 使用 softmax 处理相似度，避免直接计算相似度和对数
    exp_positive_sim = torch.exp(positive_sim / temperature)
    exp_negative_sim = torch.exp(negative_sim / temperature)

    # 正样本损失：最大化同类节点的相似度
    positive_loss = -torch.log(exp_positive_sim.sum(dim=1) + 1e-8)  # 防止数值稳定性问题

    # 负样本损失：最小化不同类节点的相似度
    negative_loss = torch.log(exp_negative_sim.sum(dim=1) + 1e-8)  # 防止数值稳定性问题

    # 总损失：结合正负样本损失
    loss = (positive_loss + negative_loss).mean()

    return loss

def get_Q(self,z):
    """计算软聚类分配矩阵Q

    Args:
        z (torch.Tensor): 节点嵌入

    Returns:
        torch.Tensor: 软分配矩阵
    """
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
    q = q.pow((1 + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q):
    """计算目标分布 P"""
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()