import torch
import torch.nn as nn
from models.BaseModel.timexer import TimeXerFeatureExtractor
from models.BaseModel.gcn import GCN

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

class UnifiedTimexerGCN(nn.Module):
    def __init__(self, 
                 configs, 
                 gcn_hidden_dim, 
                 gcn_output_dim,
                 # --- Optional Components ---
                 use_gcn: bool = True,
                 news_feature_dim: int = None, 
                 news_processed_dim: int = 32, 
                 # --- MLP and Output ---
                 mlp_hidden_dim_1: int = 256,
                 mlp_hidden_dim_2: int = 128,
                 num_classes: int = 2):
        super(UnifiedTimexerGCN, self).__init__()
        
        # --- Store configurations ---
        self.n_vars = configs.enc_in
        self.task_type = configs.task_type
        self.use_gcn = use_gcn
        self.has_news = news_feature_dim is not None and news_feature_dim > 0

        # --- Base Feature Extractor ---
        self.timexer = TimeXerFeatureExtractor(configs)
        
        # --- Dynamic Dimension Calculation ---
        # Start with the output dimension of the TimeXer feature extractor
        features_dim = configs.d_model
        
        # --- Optional News Processor ---
        if self.has_news:
            self.news_processor = MLPBlock(
                input_dim=news_feature_dim,
                hidden_dim=news_processed_dim * 2,
                output_dim=news_processed_dim,
                dropout=configs.dropout if hasattr(configs, 'dropout') else 0.3
            )
            # Add news dimension to the feature vector
            features_dim += news_processed_dim
        else:
            self.news_processor = None
            
        # --- Optional GCN ---
        if self.use_gcn:
            self.gcn = GCN(features_dim, gcn_hidden_dim, gcn_output_dim)
            mlp_input_dim = gcn_output_dim # MLP takes output from GCN
        else:
            self.gcn = None
            mlp_input_dim = features_dim # MLP takes features directly
            
        # --- Final MLP Head ---
        if self.task_type == 'regression':
            final_layer_output_dim = 1 # Predict one continuous value
        else: # 'classification'
            final_layer_output_dim = num_classes

        self.mlp = MLPBlock(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim_1,
            output_dim=final_layer_output_dim,
            dropout=configs.dropout if hasattr(configs, 'dropout') else 0.3
        )

    def forward(self, x_enc, x_mark_enc, edge_index=None, edge_weight=None, news_features=None):
        B, _, N= x_enc.shape
        
        # 1. TimeXer feature extraction
        enc_out = self.timexer(x_enc, x_mark_enc) # -> [B, N, d_model, patch_num]
        timexer_features = enc_out.mean(dim=-1)  # -> [B, N, d_model]

        # 2. News feature processing and fusion
        if self.has_news:
            if news_features is None:
                raise ValueError("Model initialized with news features, but none were provided.")
            processed_news = self.news_processor(news_features) # -> [B, N, news_processed_dim]
            fused_features = torch.cat((timexer_features, processed_news), dim=-1)
        else:
            fused_features = timexer_features

        # 3. GCN processing (optional)
        if self.use_gcn:
            if edge_index is None:
                raise ValueError("Model initialized with GCN, but no edge_index was provided.")
            features_flat = fused_features.view(B * N, -1)
            edge_index_batch = torch.cat([edge_index + i * N for i in range(B)], dim=1)

            # 处理边权重
            if edge_weight is not None:
                edge_weight_batch = edge_weight.repeat(B)
                features_for_mlp = self.gcn(features_flat, edge_index_batch, edge_weight_batch)
            else:
                features_for_mlp = self.gcn(features_flat, edge_index_batch)
        else:
            features_for_mlp = fused_features.view(B * N, -1)

        # 4. Final MLP prediction
        predictions_flat = self.mlp(features_for_mlp) # -> [B*N, output_dim]
        
        # 5. Reshape output
        if self.task_type == 'regression':
            output = predictions_flat.view(B, N) # -> [B, N]
        else: # classification
            output = predictions_flat.view(B, N, -1) # -> [B, N, num_classes]

        return output

# Example Usage to verify the unified model
if __name__ == '__main__':
    # Mock configs object
    class MockConfigs:
        def __init__(self):
            self.enc_in = 8
            self.d_model = 128
            self.patch_len = 16
            self.stride = 8
            self.num_kernels = 6
            self.d_ff = 256
            self.n_layers = 1
            self.dropout = 0.1
            self.fc_dropout = 0.1
            self.head_dropout = 0.1
            self.individual = False
            self.act = 'gelu'
            self.n_heads = 4
    
    configs = MockConfigs()
    B, N, S = 4, configs.enc_in, 180
    
    # Mock input tensors
    x_enc = torch.randn(B, N, S)
    x_mark_enc = torch.randn(B, N, S, 4) # Assuming 4 time features
    edge_list_test = [[i, j] for i in range(N) for j in range(N) if i != j]
    edge_index = torch.tensor(edge_list_test, dtype=torch.long).t().contiguous()
    news_features = torch.randn(B, N, 2315)
    
    print("--- Testing UnifiedTimexerGCN ---")

    # Test 1: Regression, No News, No GCN
    model1 = UnifiedTimexerGCN(configs, gcn_hidden_dim=64, gcn_output_dim=32, use_gcn=False, task_type='regression')
    out1 = model1(x_enc, x_mark_enc)
    assert out1.shape == (B, N)
    print("Test 1 (Regr, NoNews, NoGCN) PASSED. Output shape:", out1.shape)
    
    # Test 2: Classification, With News, No GCN
    model2 = UnifiedTimexerGCN(configs, gcn_hidden_dim=64, gcn_output_dim=32, use_gcn=False, news_feature_dim=2315, task_type='classification', num_classes=3)
    out2 = model2(x_enc, x_mark_enc, news_features=news_features)
    assert out2.shape == (B, N, 3)
    print("Test 2 (Class, News, NoGCN) PASSED. Output shape:", out2.shape)

    # Test 3: Classification, No News, With GCN
    model3 = UnifiedTimexerGCN(configs, gcn_hidden_dim=64, gcn_output_dim=32, use_gcn=True, task_type='classification', num_classes=2)
    out3 = model3(x_enc, x_mark_enc, edge_index=edge_index)
    assert out3.shape == (B, N, 2)
    print("Test 3 (Class, NoNews, GCN) PASSED. Output shape:", out3.shape)

    # Test 4: Regression, With News, With GCN (Full Model)
    model4 = UnifiedTimexerGCN(configs, gcn_hidden_dim=64, gcn_output_dim=32, use_gcn=True, news_feature_dim=2315, task_type='regression')
    out4 = model4(x_enc, x_mark_enc, edge_index=edge_index, news_features=news_features)
    assert out4.shape == (B, N)
    print("Test 4 (Regr, News, GCN) PASSED. Output shape:", out4.shape)
    
    print("\nAll tests passed successfully!") 