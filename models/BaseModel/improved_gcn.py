import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv
from torch_geometric.nn.norm import BatchNorm, GraphNorm
import math

class ImprovedGCN(nn.Module):
    """
    改进的GCN模块，增加了更多配置选项和现代化技术
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim,
                 num_layers=2,
                 dropout=0.3,
                 activation='relu',
                 use_residual=True,
                 use_batch_norm=True,
                 use_layer_norm=False,
                 conv_type='gcn',
                 heads=4,  # for GAT
                 add_self_loops=True,
                 normalize=True):
        super(ImprovedGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.conv_type = conv_type
        
        # 激活函数选择
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'elu':
            self.activation = F.elu
        else:
            self.activation = F.relu
        
        # 构建卷积层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 输入层
        if conv_type == 'gcn':
            self.convs.append(GCNConv(
                input_dim, hidden_dim, 
                add_self_loops=add_self_loops,
                normalize=normalize
            ))
        elif conv_type == 'gat':
            self.convs.append(GATConv(
                input_dim, hidden_dim // heads, 
                heads=heads, dropout=dropout,
                add_self_loops=add_self_loops
            ))
        elif conv_type == 'graph':
            self.convs.append(GraphConv(input_dim, hidden_dim))
        
        # 归一化层
        if use_batch_norm:
            self.norms.append(BatchNorm(hidden_dim))
        elif use_layer_norm:
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 隐藏层
        for i in range(num_layers - 2):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(
                    hidden_dim, hidden_dim,
                    add_self_loops=add_self_loops,
                    normalize=normalize
                ))
            elif conv_type == 'gat':
                self.convs.append(GATConv(
                    hidden_dim, hidden_dim // heads,
                    heads=heads, dropout=dropout,
                    add_self_loops=add_self_loops
                ))
            elif conv_type == 'graph':
                self.convs.append(GraphConv(hidden_dim, hidden_dim))
            
            if use_batch_norm:
                self.norms.append(BatchNorm(hidden_dim))
            elif use_layer_norm:
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 输出层
        if num_layers > 1:
            if conv_type == 'gcn':
                self.convs.append(GCNConv(
                    hidden_dim, output_dim,
                    add_self_loops=add_self_loops,
                    normalize=normalize
                ))
            elif conv_type == 'gat':
                self.convs.append(GATConv(
                    hidden_dim, output_dim,
                    heads=1, dropout=dropout,
                    add_self_loops=add_self_loops
                ))
            elif conv_type == 'graph':
                self.convs.append(GraphConv(hidden_dim, output_dim))
        
        # 残差连接的投影层
        if use_residual and input_dim != output_dim:
            self.residual_projection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_projection = None
        
        # 权重初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """改进的权重初始化"""
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        
        if self.residual_projection is not None:
            nn.init.xavier_uniform_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)
    
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
        x_input = x
        
        # 第一层
        x = self.convs[0](x, edge_index, edge_weight)
        
        if len(self.norms) > 0:
            x = self.norms[0](x)
        
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 中间层
        for i in range(1, len(self.convs) - 1):
            x_residual = x
            
            x = self.convs[i](x, edge_index, edge_weight)
            
            if i < len(self.norms):
                x = self.norms[i](x)
            
            x = self.activation(x)
            
            # 残差连接
            if self.use_residual:
                x = x + x_residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        if len(self.convs) > 1:
            x = self.convs[-1](x, edge_index, edge_weight)
        
        # 最终残差连接（输入到输出）
        if self.use_residual and self.num_layers > 1:
            if self.residual_projection is not None:
                x_input = self.residual_projection(x_input)
            if x.shape == x_input.shape:
                x = x + x_input
        
        return x

class AdaptiveGCN(nn.Module):
    """
    自适应GCN，可以根据图的特性动态调整
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim,
                 num_layers=2,
                 dropout=0.3,
                 adaptive_dropout=True,
                 adaptive_activation=True):
        super(AdaptiveGCN, self).__init__()
        
        self.num_layers = num_layers
        self.adaptive_dropout = adaptive_dropout
        self.adaptive_activation = adaptive_activation
        
        # 基础GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # 自适应组件
        if adaptive_dropout:
            # 学习每层的dropout率
            self.dropout_rates = nn.Parameter(torch.ones(num_layers) * dropout)
        else:
            # 固定dropout率，直接存储为float列表
            self.dropout_rates = [dropout] * num_layers
        
        if adaptive_activation:
            # 可学习的激活函数参数
            self.activation_params = nn.Parameter(torch.ones(num_layers))
        
        # 归一化层
        self.norms = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - 1)])
        if num_layers > 1:
            self.norms.append(BatchNorm(output_dim))
    
    def adaptive_activation_fn(self, x, layer_idx):
        """自适应激活函数"""
        if self.adaptive_activation:
            alpha = torch.sigmoid(self.activation_params[layer_idx])
            return alpha * F.relu(x) + (1 - alpha) * F.gelu(x)
        else:
            return F.relu(x)
    
    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            
            if i < len(self.norms):
                x = self.norms[i](x)
            
            if i < len(self.convs) - 1:  # 不在最后一层应用激活函数
                x = self.adaptive_activation_fn(x, i)
                
                # 自适应dropout
                if self.adaptive_dropout:
                    dropout_rate = torch.sigmoid(self.dropout_rates[i]) * 0.8  # 限制在0-0.8
                    dropout_rate = dropout_rate.item()  # 转换为float
                else:
                    dropout_rate = self.dropout_rates[i]  # 已经是float

                x = F.dropout(x, p=dropout_rate, training=self.training)
        
        return x

# 工厂函数，方便创建不同配置的GCN
def create_gcn(gcn_type='improved', **kwargs):
    """
    GCN工厂函数

    Args:
        gcn_type: 'basic', 'improved', 'adaptive'
        **kwargs: 其他参数
    """
    if gcn_type == 'basic':
        from .gcn import GCN
        return GCN(**kwargs)
    elif gcn_type == 'improved':
        return ImprovedGCN(**kwargs)
    elif gcn_type == 'adaptive':
        return AdaptiveGCN(**kwargs)
    else:
        raise ValueError(f"Unknown GCN type: {gcn_type}")

# 预设配置
GCN_CONFIGS = {
    'basic': {
        'gcn_type': 'basic',
        'description': '基础2层GCN'
    },
    'improved_light': {
        'gcn_type': 'improved',
        'num_layers': 3,
        'dropout': 0.3,
        'activation': 'relu',
        'use_residual': True,
        'use_batch_norm': True,
        'description': '轻量级改进GCN'
    },
    'improved_gelu': {
        'gcn_type': 'improved',
        'num_layers': 3,
        'dropout': 0.3,
        'activation': 'gelu',
        'use_residual': True,
        'use_batch_norm': True,
        'description': 'GELU激活改进GCN'
    },
    'gat_attention': {
        'gcn_type': 'improved',
        'num_layers': 2,
        'dropout': 0.3,
        'conv_type': 'gat',
        'heads': 4,
        'use_residual': True,
        'use_batch_norm': True,
        'description': '图注意力网络'
    },
    'adaptive': {
        'gcn_type': 'adaptive',
        'num_layers': 3,
        'dropout': 0.3,
        'adaptive_dropout': True,
        'adaptive_activation': True,
        'description': '自适应GCN'
    }
}

def create_gcn_from_config(config_name, input_dim, hidden_dim, output_dim):
    """
    从预设配置创建GCN

    Args:
        config_name: 配置名称
        input_dim, hidden_dim, output_dim: 维度参数
    """
    if config_name not in GCN_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(GCN_CONFIGS.keys())}")

    config = GCN_CONFIGS[config_name].copy()
    description = config.pop('description')

    # 添加维度参数
    config.update({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim
    })

    gcn = create_gcn(**config)
    print(f"创建GCN: {description}")
    return gcn
