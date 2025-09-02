import torch
import torch.nn as nn
from ..BaseModel.lstm import LSTMFeatureExtractor
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

class UnifiedLstmGnn(nn.Module):
    def __init__(self,
                 seq_len,
                 num_nodes,
                 input_dim=1, # price features per node
                 lstm_hidden_dim=64,
                 lstm_out_dim=32,
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

        # LSTM for price data is always present
        self.lstm_extractor = LSTMFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            out_dim=lstm_out_dim
        )
        
        # This variable will hold the dimension of features right before the GCN or final MLP
        features_pre_gcn_dim = lstm_out_dim

        # Conditionally create the news processor
        if self.has_news:
            self.news_processor = MLPBlock(
                input_dim=news_feature_dim,
                hidden_dim=news_processed_dim * 2,
                output_dim=news_processed_dim
            )
            features_pre_gcn_dim += news_processed_dim
        else:
            self.news_processor = None
        
        # This variable will hold the input dimension for the final MLP
        final_mlp_input_dim = 0

        # Conditionally create the GCN
        if self.use_gcn:
            # 使用改进的GCN配置
            print(f"🔧 LSTM使用GCN配置: {self.gcn_config}")
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
            self.output_scaler = MLPBlock(
                input_dim=1,
                hidden_dim=32,
                output_dim=1
            )

    def forward(self, price_data_x, edge_index, edge_weight=None, news_features=None):
        """
        Args:
            price_data_x: [batch, seq_len, num_nodes] or [batch, seq_len, num_nodes, input_dim]
            edge_index: [2, num_edges]
            news_features: [batch, num_nodes, news_feature_dim]
        Returns:
            Logits or regression values.
        """
        B, N = price_data_x.shape[0], self.num_nodes
        
        # 1. LSTM Feature Extraction
        # The LSTMFeatureExtractor should handle different input dims
        lstm_node_features = self.lstm_extractor(price_data_x) # Expected output: [B, N, lstm_out_dim]

        # 2. News Feature Fusion (if enabled)
        if self.has_news:
            if news_features is None:
                raise ValueError("Model initialized with news features, but none were provided in forward pass.")
            processed_news = self.news_processor(news_features) # Expected input: [B, N, news_feature_dim]
            fused_features = torch.cat((lstm_node_features, processed_news), dim=-1)
        else:
            fused_features = lstm_node_features

        # 3. GCN Processing (if enabled)
        if self.use_gcn:
            if edge_index is None:
                raise ValueError("Model initialized with GCN, but no edge_index was provided in forward pass.")

            fused_flat = fused_features.view(B * N, -1)

            # Create batched edge_index
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
    SEQ_LEN_TEST = 60
    NUM_NODES_TEST = 8
    INPUT_DIM_TEST = 1
    LSTM_HIDDEN_DIM_TEST = 64
    LSTM_OUT_DIM_TEST = 32
    GCN_HIDDEN_DIM_TEST = 128
    GCN_OUTPUT_DIM_TEST = 64
    FINAL_MLP_HIDDEN_DIM_TEST = 128
    NUM_CLASSES_TEST = 2
    BATCH_SIZE_TEST = 4
    NEWS_FEATURE_DIM_TEST = 50

    # Dummy data
    dummy_edge_index_test = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    dummy_price_data = torch.randn(BATCH_SIZE_TEST, SEQ_LEN_TEST, NUM_NODES_TEST, INPUT_DIM_TEST)
    dummy_news_features = torch.randn(BATCH_SIZE_TEST, NUM_NODES_TEST, NEWS_FEATURE_DIM_TEST)

    # --- Test Case 1: LSTM only (No News, No GCN) ---
    print("\n--- Test Case 1: LSTM only ---")
    model_lstm_only = UnifiedLstmGnn(
        seq_len=SEQ_LEN_TEST,
        num_nodes=NUM_NODES_TEST,
        input_dim=INPUT_DIM_TEST,
        use_gcn=False,
        news_feature_dim=None
    )
    print(model_lstm_only)
    output = model_lstm_only(dummy_price_data, edge_index=None)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 1 (LSTM only) PASSED.")

    # --- Test Case 2: LSTM + News (No GCN) ---
    print("\n--- Test Case 2: LSTM + News ---")
    model_lstm_news = UnifiedLstmGnn(
        seq_len=SEQ_LEN_TEST,
        num_nodes=NUM_NODES_TEST,
        input_dim=INPUT_DIM_TEST,
        use_gcn=False,
        news_feature_dim=NEWS_FEATURE_DIM_TEST
    )
    print(model_lstm_news)
    output = model_lstm_news(dummy_price_data, edge_index=None, news_features=dummy_news_features)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 2 (LSTM + News) PASSED.")

    # --- Test Case 3: LSTM + GCN (No News) ---
    print("\n--- Test Case 3: LSTM + GCN ---")
    model_lstm_gcn = UnifiedLstmGnn(
        seq_len=SEQ_LEN_TEST,
        num_nodes=NUM_NODES_TEST,
        input_dim=INPUT_DIM_TEST,
        use_gcn=True,
        news_feature_dim=None
    )
    print(model_lstm_gcn)
    output = model_lstm_gcn(dummy_price_data, edge_index=dummy_edge_index_test)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 3 (LSTM + GCN) PASSED.")

    # --- Test Case 4: Full Model (LSTM + GCN + News) ---
    print("\n--- Test Case 4: Full Model ---")
    model_full = UnifiedLstmGnn(
        seq_len=SEQ_LEN_TEST,
        num_nodes=NUM_NODES_TEST,
        input_dim=INPUT_DIM_TEST,
        use_gcn=True,
        news_feature_dim=NEWS_FEATURE_DIM_TEST
    )
    print(model_full)
    output = model_full(dummy_price_data, edge_index=dummy_edge_index_test, news_features=dummy_news_features)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST, NUM_CLASSES_TEST)
    print("Test case 4 (Full Model) PASSED.")

    # --- Test Case 5: Regression Task ---
    print("\n--- Test Case 5: Regression Task ---")
    model_regr = UnifiedLstmGnn(
        seq_len=SEQ_LEN_TEST,
        num_nodes=NUM_NODES_TEST,
        input_dim=INPUT_DIM_TEST,
        use_gcn=True,
        news_feature_dim=NEWS_FEATURE_DIM_TEST,
        task_type='regression'
    )
    print(model_regr)
    output = model_regr(dummy_price_data, edge_index=dummy_edge_index_test, news_features=dummy_news_features)
    assert output.shape == (BATCH_SIZE_TEST, NUM_NODES_TEST)
    print(f"Output shape (regression): {output.shape} - PASSED")

    print("\n--- All model tests passed! ---") 