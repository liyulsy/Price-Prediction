import torch
import torch.nn as nn

class LSTMResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 如果输入维度不等于隐藏维度，需要一个线性映射来做残差连接
        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, hidden_dim)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = self.shortcut(x)
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = out + residual
        return out

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, out_dim=32):
        super().__init__()
        
        # 初始的特征投影，将输入维度映射到hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 堆叠多个LSTM残差块
        self.lstm_blocks = nn.ModuleList([
            LSTMResidualBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 特征聚合层
        self.feature_aggregation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, out_dim)
        )
        
        # 输出缩放层，用于扩大输出范围
        self.output_scaler = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(out_dim * 2),
            nn.Linear(out_dim * 2, out_dim)
        )

    def forward(self, x):
        """
        批量处理所有节点的序列数据
        Args:
            x: [batch, seq_len, num_nodes] 或 [batch, seq_len, num_nodes, input_dim]
        Returns:
            outputs: [batch, num_nodes, out_dim]
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)  # [batch, seq_len, num_nodes, 1]
        
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # 调整维度顺序，将num_nodes移到batch维度
        x = x.permute(0, 2, 1, 3)  # [batch, num_nodes, seq_len, input_dim]
        x = x.reshape(-1, seq_len, input_dim)  # [batch * num_nodes, seq_len, input_dim]
        
        # 初始特征投影
        x = self.input_proj(x)  # [batch * num_nodes, seq_len, hidden_dim]
        
        # 通过LSTM残差块
        for lstm_block in self.lstm_blocks:
            x = lstm_block(x)  # [batch * num_nodes, seq_len, hidden_dim]
        
        # 只取最后一个时间步的特征
        x = x[:, -1]  # [batch * num_nodes, hidden_dim]
        
        # 特征聚合
        x = self.feature_aggregation(x)  # [batch * num_nodes, out_dim]
        
        # 输出缩放
        x = self.output_scaler(x)  # [batch * num_nodes, out_dim]
        
        # 恢复原始维度
        out = x.view(batch_size, num_nodes, -1)  # [batch, num_nodes, out_dim]
        
        return out
