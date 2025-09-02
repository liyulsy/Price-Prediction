import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import GraphNorm, BatchNorm
import math

class AdvancedGCN(nn.Module):
    """
    高级GCN模块，支持多种图卷积层、注意力机制、残差连接等
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim,
                 num_layers=3,
                 conv_type='gcn',  # 'gcn', 'gat', 'graph'
                 use_attention=True,
                 use_residual=True,
                 use_batch_norm=True,
                 dropout=0.3,
                 heads=4):  # for GAT
        super(AdvancedGCN, self).__init__()
        
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        
        # 构建卷积层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 输入层
        if conv_type == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif conv_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        elif conv_type == 'graph':
            self.convs.append(GraphConv(input_dim, hidden_dim))
        
        if use_batch_norm:
            self.norms.append(BatchNorm(hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            elif conv_type == 'graph':
                self.convs.append(GraphConv(hidden_dim, hidden_dim))
            
            if use_batch_norm:
                self.norms.append(BatchNorm(hidden_dim))
        
        # 输出层
        if conv_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, output_dim))
        elif conv_type == 'gat':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout))
        elif conv_type == 'graph':
            self.convs.append(GraphConv(hidden_dim, output_dim))
        
        # 注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 残差连接的投影层
        if use_residual and input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = None
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            batch: 批次信息 (可选)
        """
        # 保存输入用于残差连接
        if self.use_residual:
            if self.input_projection is not None:
                residual = self.input_projection(x)
            else:
                residual = x
        
        # 第一层
        if edge_weight is not None:
            x = self.convs[0](x, edge_index, edge_weight)
        else:
            x = self.convs[0](x, edge_index)
        
        if self.use_batch_norm and len(self.norms) > 0:
            x = self.norms[0](x)
        
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 残差连接（第一层）
        if self.use_residual:
            x = x + residual
        
        # 中间层
        for i in range(1, self.num_layers - 1):
            residual = x
            
            if edge_weight is not None:
                x = self.convs[i](x, edge_index, edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            
            if self.use_batch_norm and i < len(self.norms):
                x = self.norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 残差连接
            if self.use_residual:
                x = x + residual
        
        # 注意力机制（在最后一层之前）
        if self.use_attention and self.num_layers > 1:
            # 重塑为序列格式 [seq_len, batch_size, hidden_dim]
            x_att = x.unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            x_att = x_att.transpose(0, 1)  # [1, num_nodes, hidden_dim]
            
            att_out, _ = self.attention(x_att, x_att, x_att)
            att_out = att_out.transpose(0, 1).squeeze(1)  # [num_nodes, hidden_dim]
            
            x = self.attention_norm(x + att_out)
        
        # 最后一层
        if edge_weight is not None:
            x = self.convs[-1](x, edge_index, edge_weight)
        else:
            x = self.convs[-1](x, edge_index)
        
        return x

class TemporalGCN(nn.Module):
    """
    时序感知的GCN，能够处理时间序列特征
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim,
                 temporal_dim=64,
                 num_layers=3,
                 dropout=0.3):
        super(TemporalGCN, self).__init__()
        
        self.temporal_dim = temporal_dim
        
        # 时序特征提取
        self.temporal_encoder = nn.LSTM(input_dim, temporal_dim, batch_first=True, dropout=dropout)
        
        # 图卷积层
        self.gcn = AdvancedGCN(
            input_dim=temporal_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            conv_type='gat',  # 使用注意力图卷积
            use_attention=True,
            dropout=dropout
        )
        
        # 时序-图融合层
        self.fusion = nn.Sequential(
            nn.Linear(temporal_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, edge_weight=None, temporal_features=None):
        """
        Args:
            x: 当前时刻的节点特征 [num_nodes, input_dim]
            edge_index: 边索引
            edge_weight: 边权重
            temporal_features: 历史时序特征 [num_nodes, seq_len, input_dim]
        """
        # 图卷积处理当前特征
        gcn_out = self.gcn(x, edge_index, edge_weight)
        
        if temporal_features is not None:
            # 时序特征编码
            batch_size, seq_len, feature_dim = temporal_features.shape
            temporal_out, _ = self.temporal_encoder(temporal_features)
            temporal_out = temporal_out[:, -1, :]  # 取最后一个时间步
            
            # 融合时序和图特征
            combined = torch.cat([temporal_out, gcn_out], dim=-1)
            output = self.fusion(combined)
        else:
            output = gcn_out
        
        return output

class AdaptiveGraphLearning(nn.Module):
    """
    自适应图学习模块，能够学习最优的图结构
    """
    def __init__(self, num_nodes, feature_dim, hidden_dim=64):
        super(AdaptiveGraphLearning, self).__init__()
        
        self.num_nodes = num_nodes
        
        # 节点嵌入
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        
        # 特征到嵌入的映射
        self.feature_to_embedding = nn.Linear(feature_dim, hidden_dim)
        
        # 图学习网络
        self.graph_learner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.node_embeddings)
    
    def forward(self, node_features):
        """
        学习自适应图结构
        
        Args:
            node_features: [num_nodes, feature_dim]
        
        Returns:
            edge_index: 学习到的边索引
            edge_weights: 学习到的边权重
        """
        # 特征嵌入
        feature_embeddings = self.feature_to_embedding(node_features)
        
        # 结合节点嵌入和特征嵌入
        combined_embeddings = self.node_embeddings + feature_embeddings
        
        # 计算所有节点对的相似性
        edges = []
        weights = []
        
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # 拼接两个节点的嵌入
                pair_embedding = torch.cat([combined_embeddings[i], combined_embeddings[j]], dim=0)
                
                # 计算连接权重
                weight = self.graph_learner(pair_embedding).squeeze()
                
                # 只保留权重较高的边
                if weight > 0.1:  # 可调阈值
                    edges.extend([[i, j], [j, i]])
                    weights.extend([weight, weight])
        
        if not edges:
            # 如果没有边，创建最小连通图
            for i in range(self.num_nodes - 1):
                edges.extend([[i, i+1], [i+1, i]])
                weights.extend([0.1, 0.1])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weights

if __name__ == "__main__":
    # 测试代码
    print("Testing Advanced GCN modules...")
    
    # 测试参数
    num_nodes = 8
    input_dim = 64
    hidden_dim = 128
    output_dim = 32
    
    # 创建测试数据
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_weight = torch.randn(edge_index.shape[1])
    
    # 测试AdvancedGCN
    print("Testing AdvancedGCN...")
    model = AdvancedGCN(input_dim, hidden_dim, output_dim)
    out = model(x, edge_index, edge_weight)
    print(f"AdvancedGCN output shape: {out.shape}")
    
    # 测试TemporalGCN
    print("Testing TemporalGCN...")
    temporal_model = TemporalGCN(input_dim, hidden_dim, output_dim)
    temporal_features = torch.randn(num_nodes, 10, input_dim)  # 10个时间步
    temporal_out = temporal_model(x, edge_index, edge_weight, temporal_features)
    print(f"TemporalGCN output shape: {temporal_out.shape}")
    
    # 测试AdaptiveGraphLearning
    print("Testing AdaptiveGraphLearning...")
    adaptive_model = AdaptiveGraphLearning(num_nodes, input_dim)
    learned_edge_index, learned_edge_weights = adaptive_model(x)
    print(f"Learned graph - edges: {learned_edge_index.shape}, weights: {learned_edge_weights.shape}")
    
    print("All tests passed!")
