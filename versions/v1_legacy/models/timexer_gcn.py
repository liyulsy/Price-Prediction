import torch
import torch.nn as nn
from models.BaseModel.timexer import TimeXerFeatureExtractor
from models.BaseModel.gcn import GCN


class TimexerGCN(nn.Module):
    def __init__(self, configs, hidden_dim, output_dim, news_feature_dim: int, news_processed_dim: int = 32, num_classes: int = 2):
        super(TimexerGCN, self).__init__()
        self.n_vars = configs.enc_in  # Number of variables/nodes, e.g., from price data

        self.timexer = TimeXerFeatureExtractor(configs)  # Feature extractor for time series

        # MLP to process news features
        self.news_processor = nn.Sequential(
            nn.Linear(news_feature_dim, news_processed_dim * 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3), # Use dropout from configs if available
            nn.Linear(news_processed_dim * 2, news_processed_dim)
        )

        # GCN network
        # Input to GCN is concatenated features from TimeXer and News Processor
        gcn_input_dim = configs.d_model + news_processed_dim
        self.gcn = GCN(gcn_input_dim, hidden_dim, output_dim) # GCN module

        # Final MLP for prediction per node
        # Takes GCN output as input
        self.mlp = nn.Sequential(
            nn.Linear(96, 256),  # Input is GCN's output_dim
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3), # Use dropout from configs if available
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3), # Use dropout from configs if available
            nn.Linear(128, num_classes)
        )

    def forward(self, x_enc, x_mark_enc, edge_index, news_features):
        # x_enc: [Batch, NumVars/Nodes, SeqLen, FeaturesPerStep] (typical for TimeXer if features_per_step > 1)
        # or [Batch, SeqLen, NumVars/Nodes] if TimeXer handles permutation. Assuming TimeXer's input req.
        # x_mark_enc: [Batch, NumVars/Nodes, SeqLen, TimeFeatures] or similar
        # news_features: [Batch, NumVars/Nodes, NewsFeatureDim]
        # edge_index: [2, NumEdges] (for a single graph)

        B = x_enc.size(0)
        # N = self.n_vars # NumVars/Nodes, should match news_features.size(1) and x_enc.size(1) or (2)
        # Let's ensure N is consistent with news_features's node dimension
        if news_features.size(1) != self.n_vars:
            raise ValueError(f"Mismatch in number of nodes: self.n_vars ({self.n_vars}) vs news_features.size(1) ({news_features.size(1)})")
        N = self.n_vars

        # 1. TimeXer特征提取
        # enc_out shape: [B, n_vars, d_model, patch_num]
        enc_out = self.timexer(x_enc, x_mark_enc)
        # 平均池化patch维度得到节点特征
        timexer_node_features = enc_out.mean(dim=-1)  # Shape: [B, N, d_model]

        # 2. 新闻特征处理
        # news_features shape: [B, N, news_feature_dim]
        processed_news_features = self.news_processor(news_features)  # Shape: [B, N, news_processed_dim]

        # 3. 特征融合
        fused_features = torch.cat((timexer_node_features, processed_news_features), dim=-1)  # Shape: [B, N, d_model + news_processed_dim]

        # 4. 原GCN处理部分（已注释）
        # # 准备GCN输入：展平特征和构造批量edge_index
        # fused_features_flat = fused_features.view(B * N, -1)  # Shape: [B*N, d_model + news_processed_dim]
        # edge_index_batch_list = []
        # for i in range(B):
        #     offset = i * N
        #     edge_index_batch_list.append(edge_index + offset)
        # edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)  # Shape: [2, B * NumEdges]
        # # GCN前向传播
        # gcn_out_flat = self.gcn(fused_features_flat, edge_index_for_gcn)  # Shape: [B*N, gcn_output_dim (output_dim)]

        # 5. 直接使用融合特征进行预测（跳过GCN）
        fused_features_flat = fused_features.view(B * N, -1)  # Shape: [B*N, d_model + news_processed_dim]
        mlp_predictions_flat = self.mlp(fused_features_flat)  # Shape: [B*N, num_classes]

        # 6. 恢复batch结构 [B, N, num_classes]
        output = mlp_predictions_flat.view(B, N, -1)

        return output
    

    # def forward(self, x_enc, x_mark_enc, edge_index, news_features):
        # # x_enc: [Batch, NumVars/Nodes, SeqLen, FeaturesPerStep] (typical for TimeXer if features_per_step > 1)
        # # or [Batch, SeqLen, NumVars/Nodes] if TimeXer handles permutation. Assuming TimeXer's input req.
        # # x_mark_enc: [Batch, NumVars/Nodes, SeqLen, TimeFeatures] or similar
        # # news_features: [Batch, NumVars/Nodes, NewsFeatureDim]
        # # edge_index: [2, NumEdges] (for a single graph)

        # B = x_enc.size(0)
        # # N = self.n_vars # NumVars/Nodes, should match news_features.size(1) and x_enc.size(1) or (2)
        # # Let's ensure N is consistent with news_features's node dimension
        # if news_features.size(1) != self.n_vars:
        #     raise ValueError(f"Mismatch in number of nodes: self.n_vars ({self.n_vars}) vs news_features.size(1) ({news_features.size(1)})")
        # N = self.n_vars

        # # 1. TimeXer feature extraction
        # # enc_out shape: [B, n_vars, d_model, patch_num]
        # enc_out = self.timexer(x_enc, x_mark_enc)
        # # Average pool over patch dimension to get node features
        # timexer_node_features = enc_out.mean(dim=-1)  # Shape: [B, N, d_model]

        # # 2. News feature processing
        # # news_features shape: [B, N, news_feature_dim]
        # processed_news_features = self.news_processor(news_features)  # Shape: [B, N, news_processed_dim]

        # # 3. Feature fusion
        # fused_features = torch.cat((timexer_node_features, processed_news_features), dim=-1)  # Shape: [B, N, d_model + news_processed_dim]

        # # 4. Prepare for GCN: Reshape and batch edge_index
        # fused_features_flat = fused_features.view(B * N, -1)  # Shape: [B*N, d_model + news_processed_dim]

        # edge_index_batch_list = []
        # for i in range(B):
        #     offset = i * N
        #     edge_index_batch_list.append(edge_index + offset)
        # edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)  # Shape: [2, B * NumEdges]

        # # 5. GCN forward pass
        # gcn_out_flat = self.gcn(fused_features_flat, edge_index_for_gcn)  # Shape: [B*N, gcn_output_dim (output_dim)]

        # # 6. MLP forward pass
        # # gcn_out_flat is already in the correct shape [B*N, output_dim] for the MLP with BatchNorm1d
        # mlp_predictions_flat = self.mlp(gcn_out_flat)  # Shape: [B*N, num_classes]

        # # 7. Reshape output to [B, N, num_classes]
        # output = mlp_predictions_flat.view(B, N, -1)

        # return output


