import torch
import torch.nn as nn

class CnnResidualBlock(nn.Module):
    """
    一个包含两个1D卷积层和残差连接的块。
    这使得网络可以构建得更深，从而学习更复杂的模式。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        # 为了保持维度不变，padding = (kernel_size - 1) / 2
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入和输出的通道数不同，或者步长不为1，
        # 我们需要一个额外的1x1卷积来调整残差(shortcut)的维度，以确保它们可以相加。
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # 残差连接
        out = self.relu2(out)
        return out

class CNN(nn.Module):
    """
    一个基于残差块的一维卷积网络，用于从价格序列中提取特征。
    这个实现比之前的版本强大得多。
    """
    def __init__(self, in_dim, num_nodes, out_channels=32, num_blocks=2, kernel_size=7):
        """
        Args:
            in_dim (int): 输入的序列长度。这个参数在此实现中不直接使用，但为了保持接口一致性而保留。
            num_nodes (int): 节点的数量。
            out_channels (int): CNN最终输出的特征维度。
            num_blocks (int): 要堆叠的残差块的数量。
            kernel_size (int): 卷积核的大小。
        """
        super().__init__()
        self.num_nodes = num_nodes
        
        # 初始卷积层，将输入通道从1（仅价格）增加到out_channels
        initial_channels = 16
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, initial_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(initial_channels),
            nn.ReLU()
        )

        # 堆叠多个残差块
        blocks = []
        current_channels = initial_channels
        for _ in range(num_blocks):
            blocks.append(CnnResidualBlock(current_channels, current_channels, kernel_size=kernel_size))
        
        self.residual_blocks = nn.Sequential(*blocks)
        
        # 全局平均池化，将时间维度上的信息聚合起来
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 最后的线性层，产生最终的输出特征
        self.final_fc = nn.Linear(current_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 价格数据，形状为 [Batch, Seq_Len, Num_Nodes]
        
        Returns:
            torch.Tensor: 提取的节点特征，形状为 [Batch, Num_nodes, out_channels]
        """
        B, L, N = x.shape
        
        # 我们需要对每个节点的时间序列独立应用CNN。
        # 1. 调整维度以匹配Conv1d的期望输入：[Batch, Channels, Length]
        # [B, L, N] -> [B, N, L] -> [B * N, 1, L]
        x_reshaped = x.permute(0, 2, 1).reshape(B * N, 1, L)
        
        # 2. 通过初始卷积层
        # [B * N, 1, L] -> [B * N, initial_channels, L]
        x_out = self.input_conv(x_reshaped)
        
        # 3. 通过残差块
        # [B * N, initial_channels, L] -> [B * N, current_channels, L]
        x_out = self.residual_blocks(x_out)
        
        # 4. 全局平均池化
        # [B * N, current_channels, L] -> [B * N, current_channels, 1]
        x_out = self.global_avg_pool(x_out)
        
        # 5. 展平并应用最后的线性层
        # [B * N, current_channels, 1] -> [B * N, current_channels]
        x_out = x_out.squeeze(-1)
        # [B * N, current_channels] -> [B * N, out_channels]
        x_out = self.final_fc(x_out)
        
        # 6. 恢复原始的Batch和节点维度
        # [B * N, out_channels] -> [B, N, out_channels]
        node_features = x_out.view(B, N, -1)
        
        return node_features