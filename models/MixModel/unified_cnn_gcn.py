import torch
import torch.nn as nn
from ..BaseModel.cnn import CNN 
from ..BaseModel.gcn import GCN
from models.BaseModel.improved_gcn import create_gcn_from_config

class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):  
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 初始化权重，特别是最后一层
        self._init_weights()
        
    def forward(self, x):
        # First layer
        h1 = self.fc1(x)
        h1 = torch.relu(h1)
        h1 = self.layer_norm1(h1)
        h1 = self.dropout(h1)
        
        # Second layer with residual connection
        h2 = self.fc2(h1)
        h2 = torch.relu(h2)
        h2 = self.layer_norm2(h2)
        h2 = self.dropout(h2)
        h2 = h2 + h1  # Residual connection
        
        # Output layer
        out = self.fc3(h2)
        return out

    def _init_weights(self):
        """初始化权重，避免偏向某一类别"""
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化
                nn.init.xavier_normal_(m.weight, gain=0.5)  # 减小初始权重
                if m.bias is not None:
                    # 最后一层的偏置初始化为小的随机值
                    if i == len(list(self.modules())) - 1:  # 最后一层
                        nn.init.uniform_(m.bias, -0.01, 0.01)
                    else:
                        nn.init.constant_(m.bias, 0)

class UnifiedCnnGnn(nn.Module):
    def __init__(self,
                 price_seq_len,
                 num_nodes,
                 cnn_output_channels=32,
                 gcn_hidden_dim=128,
                 gcn_output_dim=64,
                 final_mlp_hidden_dim=128,
                 num_classes=2,
                 # Optional components switches
                 task_type: str = 'classification', # 'classification' or 'regression'
                 use_gcn: bool = True,
                 gcn_config: str = 'improved_gelu',  # 新增：GCN配置选择
                 news_feature_dim=None,
                 news_processed_dim=32):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.task_type = task_type
        self.use_gcn = use_gcn
        self.gcn_config = gcn_config
        self.has_news = news_feature_dim is not None and news_feature_dim > 0

        # CNN for price data is always present
        self.price_cnn = CNN(in_dim=price_seq_len, num_nodes=num_nodes, out_channels=cnn_output_channels)
        
        # This variable will hold the dimension of features right before the GCN or final MLP
        features_pre_gcn_dim = cnn_output_channels

        # Conditionally create the news processor
        if self.has_news:
            # 添加新闻特征归一化
            self.news_norm = nn.LayerNorm(news_feature_dim)
            self.news_processor = MLPBlock(
                input_dim=news_feature_dim,
                hidden_dim=news_processed_dim * 2,
                output_dim=news_processed_dim
            )
            # 添加融合后的归一化
            self.fusion_norm = nn.LayerNorm(cnn_output_channels + news_processed_dim)
            # 添加可学习的特征权重
            self.feature_gate = nn.Parameter(torch.tensor(0.5))  # 初始化为0.5，平衡两种特征
            features_pre_gcn_dim += news_processed_dim
        
        # This variable will hold the input dimension for the final MLP
        final_mlp_input_dim = 0

        # Conditionally create the GCN
        if self.use_gcn:
            # 使用改进的GCN配置
            print(f"🔧 CNN使用GCN配置: {self.gcn_config}")
            self.gcn = create_gcn_from_config(self.gcn_config, features_pre_gcn_dim, gcn_hidden_dim, gcn_output_dim)
            final_mlp_input_dim = gcn_output_dim
        else:
            self.gcn = None
            final_mlp_input_dim = features_pre_gcn_dim

        # Final MLP's output dimension depends on the task type
        if self.task_type == 'regression':
            final_layer_output_dim = 1
        else:
            final_layer_output_dim = num_classes

        # Enhanced MLP with residual connections and layer normalization
        self.mlp = MLPBlock(
            input_dim=final_mlp_input_dim,
            hidden_dim=final_mlp_hidden_dim,
            output_dim=final_layer_output_dim
        )

        # For regression, add a scaling layer to help with output range
        if self.task_type == 'regression':
            self.output_scaler = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, price_data_x, edge_index, edge_weight=None, news_features=None, news_weight_scale=1.0):
        # 1. CNN Feature Extraction
        cnn_node_features = self.price_cnn(price_data_x)

        # 2. News Feature Fusion (if enabled)
        if self.has_news:
            if news_features is None:
                raise ValueError("Model initialized with news features, but none were provided in forward pass.")
            # 归一化新闻特征
            normalized_news = self.news_norm(news_features)
            processed_news = self.news_processor(normalized_news)

            # 使用可学习权重平衡特征，并应用动态缩放
            gate = torch.sigmoid(self.feature_gate)  # 确保权重在0-1之间
            weighted_cnn = cnn_node_features * gate
            # 应用动态新闻权重缩放
            weighted_news = processed_news * (1 - gate) * news_weight_scale

            fused_features = torch.cat((weighted_cnn, weighted_news), dim=-1)
            # 归一化融合后的特征
            fused_features = self.fusion_norm(fused_features)
        else:
            fused_features = cnn_node_features

        # 3. GCN Processing (if enabled)
        if self.use_gcn:
            if edge_index is None:
                raise ValueError("Model initialized with GCN, but no edge_index was provided in forward pass.")
            B, N, _ = fused_features.shape
            fused_flat = fused_features.view(B * N, -1)

            edge_index_batch = torch.cat([edge_index + i * N for i in range(B)], dim=1)

            # 处理边权重
            if edge_weight is not None:
                edge_weight_batch = edge_weight.repeat(B)
                gcn_out_flat = self.gcn(fused_flat, edge_index_batch, edge_weight_batch)
            else:
                gcn_out_flat = self.gcn(fused_flat, edge_index_batch)
            features_for_mlp = gcn_out_flat.view(B, N, -1)
        else:
            features_for_mlp = fused_features
        
        # 4. Final Prediction MLP
        out = self.mlp(features_for_mlp)

        # For regression, apply additional scaling to expand the output range
        if self.task_type == 'regression':
            out = self.output_scaler(out)
            return out.squeeze(-1)
        return out

if __name__ == '__main__':
    # Common test parameters
    PRICE_SEQ_LEN_TEST = 180      
    NUM_NODES_TEST = 8            
    CNN_OUT_CHANNELS_TEST = 32
    GCN_HIDDEN_DIM_TEST = 128
    GCN_OUTPUT_DIM_TEST = 64      
    FINAL_MLP_HIDDEN_DIM_TEST = 128
    NUM_CLASSES_TEST = 2
    BATCH_SIZE_TEST = 4

    # Dummy edge index
    edge_list_test = [[i, j] for i in range(NUM_NODES_TEST) for j in range(NUM_NODES_TEST) if i != j]
    dummy_edge_index_test = torch.tensor(edge_list_test, dtype=torch.long).t().contiguous()
    dummy_price_data = torch.randn(BATCH_SIZE_TEST, PRICE_SEQ_LEN_TEST, NUM_NODES_TEST)

    # --- Test Case 1: CNN only (No News, No GCN) ---
    print("\n--- Test Case 1: CNN only ---")
    model_cnn_only = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN_TEST, num_nodes=NUM_NODES_TEST,
        use_gcn=False, news_feature_dim=None
    )
    print(model_cnn_only)
    output = model_cnn_only(dummy_price_data, edge_index=None)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 1 (CNN only) PASSED.")

    # --- Test Case 2: CNN + News (No GCN) ---
    print("\n--- Test Case 2: CNN + News ---")
    NEWS_FEATURE_DIM_TEST = 2315
    dummy_news_features = torch.randn(BATCH_SIZE_TEST, NUM_NODES_TEST, NEWS_FEATURE_DIM_TEST)
    model_cnn_news = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN_TEST, num_nodes=NUM_NODES_TEST,
        use_gcn=False, news_feature_dim=NEWS_FEATURE_DIM_TEST
    )
    print(model_cnn_news)
    output = model_cnn_news(dummy_price_data, edge_index=None, news_features=dummy_news_features)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 2 (CNN + News) PASSED.")

    # --- Test Case 3: CNN + GCN (No News) ---
    print("\n--- Test Case 3: CNN + GCN ---")
    model_cnn_gcn = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN_TEST, num_nodes=NUM_NODES_TEST,
        use_gcn=True, news_feature_dim=None
    )
    print(model_cnn_gcn)
    output = model_cnn_gcn(dummy_price_data, edge_index=dummy_edge_index_test)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 3 (CNN + GCN) PASSED.")
    
    # --- Test Case 4: Full Model (CNN + GCN + News) ---
    print("\n--- Test Case 4: Full Model ---")
    model_full = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN_TEST, num_nodes=NUM_NODES_TEST,
        use_gcn=True, news_feature_dim=NEWS_FEATURE_DIM_TEST
    )
    print(model_full)
    output = model_full(dummy_price_data, edge_index=dummy_edge_index_test, news_features=dummy_news_features)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 4 (Full Model) PASSED.")

    # --- Test Case 1: Regression (CNN + GCN, no news) ---
    print("\n--- Test Case 1: Regression (CNN + GCN) ---")
    model_regr = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN_TEST, num_nodes=NUM_NODES_TEST,
        use_gcn=True, news_feature_dim=None, task_type='regression'
    )
    print(model_regr)
    output = model_regr(dummy_price_data, edge_index=dummy_edge_index_test)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST)
    print(f"Output shape (regression): {output.shape} - PASSED")

    # --- Test Case 2: Classification (Full Model) ---
    print("\n--- Test Case 2: Classification (Full Model) ---")
    model_class = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN_TEST, num_nodes=NUM_NODES_TEST,
        use_gcn=True, news_feature_dim=NEWS_FEATURE_DIM_TEST, task_type='classification',
        num_classes=NUM_CLASSES_TEST
    )
    print(model_class)
    output = model_class(dummy_price_data, dummy_edge_index_test, news_features=dummy_news_features)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print(f"Output shape (classification): {output.shape} - PASSED")
    
    print("\n--- All model tests passed! ---") 