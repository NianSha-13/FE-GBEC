from torch_geometric.nn import GCNConv
import warnings

import torch
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv, GATConv

# # import matplotlib.pyplot as plt
# from LWEA_codes.computeECI import computeECI
# from LWEA_codes.computeLWCA import computeLWCA
# from LWEA_codes.getAllSegs import getAllSegs
# from getCA import getCA
# from getHC import getHC
# from run_EC_CMSToALL import run_EC_CMS

warnings.filterwarnings('ignore')
import numpy as np
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import torch.nn.functional as F


alpha = 0.8  # Initialization parameter alpha
_lambda = 0.4


class Model(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels  # 每个节点的特征维度
        self.out_channels = out_channels  # 输出特征维度
        self.GCN_layer = GCN(in_channels=out_channels, out_channels=out_channels)
        self.Linear1 = nn.Linear(out_channels, out_channels)
        self.GAT_layer = GATModel(in_channels, 16, out_channels, heads=3)
        #self.Linear2 = nn.Linear(64, out_channels)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = self.GAT_layer(x, edge_index)
        x = self.GCN_layer(x, edge_index)
        x = self.Linear1(x)
        #x = self.Linear2(x)
        return x



class GCN(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)  # 第一层图卷积
        self.conv2 = GCNConv(16, out_channels)  # 第二层图卷积
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # 第一层卷积 + ReLU 激活
        # x = self.conv2(x, edge_index)  # 第二层卷积
        # x = F.relu(x)
        return x

class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super(GATLayer, self).__init__()
        # GATConv 是 PyTorch Geometric 提供的图注意力卷积层
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, x, edge_index):
        # x 是节点的特征，edge_index 是图的边索引
        return self.gat_conv(x, edge_index)

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GATModel, self).__init__()
        # 第一层 GAT
        self.gat1 = GATLayer(in_channels, hidden_channels, heads=heads)
        # 第二层 GAT
        self.gat2 = GATLayer(hidden_channels * heads, out_channels, heads=1)  # 只需要一个输出头

    def forward(self, x, edge_index):
        # x 是节点的特征，edge_index 是边的索引
        x = F.elu(self.gat1(x, edge_index))  # 第一个 GAT 层
        x = self.gat2(x, edge_index)  # 第二个 GAT 层
        return F.log_softmax(x, dim=1)  # 使用 softmax 激活输出