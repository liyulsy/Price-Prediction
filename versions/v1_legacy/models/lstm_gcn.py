import torch
import torch.nn as nn
from models.BaseModel.gcn import GCN
from models.BaseModel.lstm import LSTMFeatureExtractor

class LstmGcn(nn.Module):
    def __init__(self, 
                 seq_len,           # 输入序列长度
                 num_nodes,         # 节点数量
                 input_dim,         # 每个时间步的特征维度（例如：1表示只用收盘价）
                 news_feature_dim=None,  # 新闻特征原始维度，None表示不使用新闻
                 lstm_hidden_dim=64,  # LSTM隐藏层维度
                 lstm_out_dim=32,     # LSTM输出维度
                 news_processed_dim=32, # 处理后的新闻特征维度
                 gcn_hidden_dim=64,   # GCN隐藏层维度
                 gcn_output_dim=32,   # GCN输出维度
                 mlp_hidden_dim=128,  # MLP隐藏层维度
                 num_classes=2,       # 输出类别数
                 use_gcn=False,       # 是否使用GCN
                 use_news=False):     # 是否使用新闻特征
        super(LstmGcn, self).__init__()
        
        self.num_nodes = num_nodes
        self.use_gcn = use_gcn
        self.use_news = use_news and news_feature_dim is not None  # 确保有新闻特征维度时才能使用新闻
        
        # 确保 lstm_out_dim 和 news_processed_dim 相同，以便后续特征拼接
        if self.use_news and lstm_out_dim != news_processed_dim:
            print(f"Warning: lstm_out_dim ({lstm_out_dim}) != news_processed_dim ({news_processed_dim})")
            print("Adjusting news_processed_dim to match lstm_out_dim")
            news_processed_dim = lstm_out_dim
        
        # LSTM特征提取器
        self.lstm = LSTMFeatureExtractor(
            input_dim=input_dim, 
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim
        )
        
        # 新闻特征处理MLP（仅当use_news=True时使用）
        if self.use_news:
            self.news_processor = nn.Sequential(
                nn.Linear(news_feature_dim, news_processed_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(news_processed_dim * 2, news_processed_dim)
            )
            combined_feature_dim = lstm_out_dim + news_processed_dim
        else:
            self.news_processor = None
            combined_feature_dim = lstm_out_dim

        # GCN网络
        self.gcn = GCN(
            input_dim=combined_feature_dim,  # LSTM特征 + 新闻特征（如果使用）
            hidden_dim=gcn_hidden_dim,
            output_dim=gcn_output_dim
        )
        
        # MLP网络 - 输入维度根据是否使用GCN和新闻来决定
        mlp_input_dim = gcn_output_dim if use_gcn else combined_feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_dim, num_classes)
        )
        
    def forward(self, x, edge_index, news_features=None):
        """
        前向传播，使用批量处理提高速度
        Args:
            x: [batch, seq_len, num_nodes] 或 [batch, seq_len, num_nodes, input_dim]
            edge_index: [2, num_edges] 图的边信息
            news_features: [batch, num_nodes, news_feature_dim] 新闻特征（如果use_news=True）
        Returns:
            logits: [batch, num_nodes, num_classes] 每个节点的预测结果
        """
        batch_size = x.size(0)
        
        # 1. LSTM特征提取 - 已优化为批量处理
        lstm_features = self.lstm(x)  # [batch, num_nodes, lstm_out_dim]
        
        # 2. 特征准备
        if self.use_news:
            if news_features is None:
                raise ValueError("News features are required when use_news=True")
            
            # 批量处理新闻特征
            news_flat = news_features.view(-1, news_features.size(-1))  # [batch * num_nodes, news_feature_dim]
            processed_news = self.news_processor(news_flat)  # [batch * num_nodes, news_processed_dim]
            processed_news = processed_news.view(batch_size, self.num_nodes, -1)  # [batch, num_nodes, news_processed_dim]
            
            # 特征融合
            features = torch.cat([lstm_features, processed_news], dim=-1)  # [batch, num_nodes, combined_feature_dim]
        else:
            features = lstm_features
        
        # 3. GCN处理（如果启用）
        if self.use_gcn:
            # 合并batch和节点维度
            features_flat = features.reshape(-1, features.size(-1))  # [batch * num_nodes, feature_dim]
            
            # 批量处理edge_index
            edge_index_batch = torch.cat([edge_index + i * self.num_nodes for i in range(batch_size)], dim=1)
            
            # GCN前向传播
            gcn_out = self.gcn(features_flat, edge_index_batch)  # [batch * num_nodes, gcn_output_dim]
            final_features = gcn_out.view(batch_size, self.num_nodes, -1)  # [batch, num_nodes, gcn_output_dim]
        else:
            final_features = features
        
        # 4. MLP预测 - 批量处理
        final_flat = final_features.view(-1, final_features.size(-1))  # [batch * num_nodes, feature_dim]
        logits_flat = self.mlp(final_flat)  # [batch * num_nodes, num_classes]
        logits = logits_flat.view(batch_size, self.num_nodes, -1)  # [batch, num_nodes, num_classes]
        
        return logits
