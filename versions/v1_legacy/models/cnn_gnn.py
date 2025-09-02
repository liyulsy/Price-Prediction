import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.BaseModel.cnn import CNN
from models.BaseModel.gcn import GCN
import pandas as pd

class CnnGnn(nn.Module):
    def __init__(self, 
                 price_seq_len,      # Input to CNN (sequence length per node)
                 num_nodes,          # Number of nodes (coins)
                 news_feature_dim,   # Raw dimension of news features per node
                 gcn_hidden_dim,     # Hidden dimension for GCN
                 gcn_output_dim,     # Output dimension from GCN (input to final MLP)
                 cnn_output_channels, # Output dimension from CNN (input to GCN)
                 news_processed_dim=32, # Dimension of news features after processing MLP
                 final_mlp_hidden_dim=128, 
                 num_classes=2):
        super(CnnGnn, self).__init__()
        
        # CNN for price data - from cnn.py, it outputs 32 features per node
        self.cnn = CNN(in_dim=price_seq_len, num_nodes=num_nodes)
        _cnn_hardcoded_output_dim = cnn_output_channels # Matching the provided cnn.py

        # MLP to process news features for each node
        self.news_processor = nn.Sequential(
            nn.Linear(news_feature_dim, news_processed_dim * 2), # e.g., 2315 -> 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(news_processed_dim * 2, news_processed_dim)  # e.g., 64 -> 32
        )

        # GCN network for node features after fusion
        # Input to GCN is concatenated features from CNN and News Processor
        gcn_input_dim = _cnn_hardcoded_output_dim + news_processed_dim # e.g., 32 + 32 = 64
        self.gcn = GCN(input_dim=gcn_input_dim, 
                       hidden_dim=gcn_hidden_dim, 
                       output_dim=gcn_output_dim)
        
        # Final MLP for prediction per node
        self.mlp = nn.Sequential(
            nn.Linear(gcn_output_dim, final_mlp_hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(final_mlp_hidden_dim, num_classes) 
        )
        
    # def forward(self, price_data_x, edge_index, news_features):
    #     # 1. Prepare price data
    #     if price_data_x.dim() == 3 and price_data_x.shape[1] != self.cnn.num_nodes:
    #         price_data_x_permuted = price_data_x.permute(0, 2, 1)
    #     else:
    #         price_data_x_permuted = price_data_x

    #     # 2. CNN 特征提取
    #     cnn_node_features = self.cnn(price_data_x_permuted)  # [B, NumNodes, cnn_dim]

    #     # 3. 新闻特征处理
    #     processed_news_node_features = self.news_processor(news_features)  # [B, NumNodes, news_dim]

    #     # 4. 特征融合
    #     fused_node_features = torch.cat((cnn_node_features, processed_news_node_features), dim=-1)  # [B, NumNodes, F]

    #     # # 5. reshape 为 GCN 批处理格式
    #     # B, N, F = fused_node_features.shape
    #     # fused_node_features_flat = fused_node_features.view(B * N, F)  # [B * N, F]

    #     # # 6. 扩展 edge_index 到 batch 模式
    #     # # 每个样本图结构相同，我们需要偏移 edge_index 的节点索引
    #     # edge_index_batch = []
    #     # for i in range(B):
    #     #     offset = i * N
    #     #     edge_index_batch.append(edge_index + offset)
    #     # edge_index_batch = torch.cat(edge_index_batch, dim=1)  # [2, B * num_edges]

    #     # # 7. 批处理 GCN
    #     # gcn_out = self.gcn(fused_node_features_flat, edge_index_batch)  # [B * N, gcn_out_dim]

    #     # # 8. 恢复为 [B, N, gcn_out_dim]
    #     # gcn_out = gcn_out.view(B, N, -1)

    #     # 9. MLP 预测 (直接使用融合特征)
    #     out = self.mlp(fused_node_features)  # [B, N, num_classes]

    #     return out
    
    def forward(self, price_data_x, edge_index, news_features):
        # 1. Prepare price data
        if price_data_x.dim() == 3 and price_data_x.shape[1] != self.cnn.num_nodes:
            price_data_x_permuted = price_data_x.permute(0, 2, 1)
        else:
            price_data_x_permuted = price_data_x

        # 2. CNN 特征提取
        cnn_node_features = self.cnn(price_data_x_permuted)  # [B, NumNodes, cnn_dim]

        # 3. 新闻特征处理
        processed_news_node_features = self.news_processor(news_features)  # [B, NumNodes, news_dim]

        # 4. 特征融合
        fused_node_features = torch.cat((cnn_node_features, processed_news_node_features), dim=-1)  # [B, NumNodes, F]

        # 5. reshape 为 GCN 批处理格式
        B, N, F = fused_node_features.shape
        fused_node_features_flat = fused_node_features.view(B * N, F)  # [B * N, F]

        # 6. 扩展 edge_index 到 batch 模式
        # 每个样本图结构相同，我们需要偏移 edge_index 的节点索引
        edge_index_batch = []
        for i in range(B):
            offset = i * N
            edge_index_batch.append(edge_index + offset)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)  # [2, B * num_edges]

        # 7. 批处理 GCN
        gcn_out = self.gcn(fused_node_features_flat, edge_index_batch)  # [B * N, gcn_out_dim]

        # 8. 恢复为 [B, N, gcn_out_dim]
        gcn_out = gcn_out.view(B, N, -1)

        # 9. MLP 预测
        out = self.mlp(gcn_out)  # [B, N, num_classes]

        return out

# Example Usage (Illustrative - ensure your BaseModel paths are correct for imports):
if __name__ == '__main__':
    PRICE_SEQ_LEN = 24       
    NUM_NODES = 8            
    NEWS_FEATURE_DIM = 2315  
    NEWS_PROCESSED_DIM = 32  
    GCN_HIDDEN_DIM = 64
    GCN_OUTPUT_DIM = 32     
    CNN_OUTPUT_CHANNELS = 32
    FINAL_MLP_HIDDEN_DIM = 128 
    NUM_CLASSES = 2
    BATCH_SIZE = 4

    model = CnnGnn(
        price_seq_len=PRICE_SEQ_LEN,
        num_nodes=NUM_NODES,
        news_feature_dim=NEWS_FEATURE_DIM,
        news_processed_dim=NEWS_PROCESSED_DIM,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        cnn_output_channels=CNN_OUTPUT_CHANNELS,
        final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES
    )
    model.train()

    dummy_price_data_x = torch.randn(BATCH_SIZE, PRICE_SEQ_LEN, NUM_NODES) # Standard: [B, SeqLen, NumNodes]
    dummy_news_features = torch.randn(BATCH_SIZE, NUM_NODES, NEWS_FEATURE_DIM)
    edge_list = []
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j:
                edge_list.append([i, j])
    dummy_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    print(f"\nFeeding dummy data to the model...")
    print(f"Dummy price_data_x (original for DataLoader): {dummy_price_data_x.shape}")
    # price_data_x_permuted_for_cnn = dummy_price_data_x.permute(0,2,1)
    # print(f"Dummy price_data_x (permuted for CNN): {price_data_x_permuted_for_cnn.shape}")
    print(f"Dummy news_features shape: {dummy_news_features.shape}")
    print(f"Dummy edge_index shape: {dummy_edge_index.shape}")

    output_logits = model(dummy_price_data_x, dummy_edge_index, dummy_news_features)

    print(f"\nOutput logits shape: {output_logits.shape}")
    print(f"Output logits (first sample, first node): {output_logits[0, 0, :]}")

