import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        # GCN 层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight=None):
        # x: 节点特征, edge_index: 边的连接信息, edge_weight: 边权重（可选）
        x = F.relu(self.conv1(x, edge_index, edge_weight))  # 第一层 GCN
        x = F.dropout(x, p=0.5, training=self.training)  # Dropout
        x = self.conv2(x, edge_index, edge_weight)  # 第二层 GCN
        return x