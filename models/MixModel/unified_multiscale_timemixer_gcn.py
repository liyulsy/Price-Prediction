import torch
import torch.nn as nn
from models.BaseModel.timemixer import TimemixerFeatureExtractor
from models.BaseModel.gcn import GCN
from models.layers.standardnorm import Normalize
from models.BaseModel.improved_gcn import create_gcn_from_config

class UnifiedMultiScaleTimeMixer(nn.Module):
    def __init__(self, configs,
                 # Optional components
                 use_gcn: bool,
                 gcn_config: str = 'improved_gelu',  # 新增：GCN配置选择
                 news_feature_dim: int = None,
                 # Dimensions
                 gcn_hidden_dim: int = 128,
                 gcn_output_dim: int = 64,
                 news_processed_dim: int = 32,
                 prediction_head_dim: int = 64,
                 mlp_hidden_dim: int = 256,
                 num_classes: int = 2):
        super(UnifiedMultiScaleTimeMixer, self).__init__()

        self.configs = configs
        self.task_type = configs.task_type
        self.use_gcn = use_gcn
        self.gcn_config = gcn_config
        self.has_news = news_feature_dim is not None and news_feature_dim > 0

        # --- Base Feature Extractor ---
        self.timemixer = TimemixerFeatureExtractor(configs)

        # --- Normalization Layers (common to both tasks) ---
        self.normalize_layers = nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
            for _ in range(configs.down_sampling_layers + 1)
        ])

        # --- Optional News Processor (common to both tasks) ---
        if self.has_news:
            self.news_processor = nn.Sequential(
                nn.Linear(news_feature_dim, news_processed_dim * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3),
                nn.Linear(news_processed_dim * 2, news_processed_dim)
            )
        else:
            self.news_processor = None

        # --- GCN/Downstream input dimension calculation ---
        downstream_feature_dim = configs.d_model
        if self.has_news:
            downstream_feature_dim += news_processed_dim

        # --- Optional GCN Layers (common to both tasks) ---
        if self.use_gcn:
            # 使用改进的GCN配置
            print(f"🔧 TimeMixer使用GCN配置: {self.gcn_config}")
            # For regression, one per scale. For classification, only the first is used.
            self.gcn_layers = nn.ModuleList([
                create_gcn_from_config(self.gcn_config, downstream_feature_dim, gcn_hidden_dim, gcn_output_dim)
                for _ in range(configs.down_sampling_layers + 1)
            ])
            # Update the feature dimension for the next layer
            downstream_feature_dim = gcn_output_dim
        else:
            self.gcn_layers = None

        # --- Task-specific Heads ---
        if self.task_type == 'classification':
            self.act = nn.GELU()
            self.dropout = nn.Dropout(0.3)
            # Input to projection is flattened features + time encoding
            projection_input_dim = configs.seq_len * (downstream_feature_dim + configs.num_time_features)
            self.projection = nn.Linear(projection_input_dim, num_classes)
            
            # These layers are not used in classification mode
            self.predict_layers = None
            self.mlp = None
        else: # Regression mode layers
            self.predict_layers = nn.ModuleList([
                nn.Linear(downstream_feature_dim, prediction_head_dim)
                for _ in range(configs.down_sampling_layers + 1)
            ])

            self.mlp = nn.Sequential(
                nn.Linear(prediction_head_dim, mlp_hidden_dim),
                nn.BatchNorm1d(mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
                nn.BatchNorm1d(mlp_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(mlp_hidden_dim // 2, num_classes)
            )

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        # This is a helper function copied from the source files.
        # It creates multi-scale versions of the input data.
        if self.configs.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                kernel_size=3, padding=padding,
                                stride=self.configs.down_sampling_window,
                                padding_mode='circular', bias=False)
        else:
            return [x_enc], [x_mark_enc] if x_mark_enc is not None else None
        
        x_enc_p = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc_p
        x_mark_enc_mark_ori = x_mark_enc
        x_enc_sampling_list = [x_enc]
        x_mark_sampling_list = [x_mark_enc]

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]
        
        return x_enc_sampling_list, x_mark_sampling_list if x_mark_enc is not None else None

    def forward(self, x_enc, x_mark_enc, edge_index=None, edge_weight=None, news_features=None):
        B, S, N = x_enc.shape
        
        if self.task_type == 'classification':
            # --- CLASSIFICATION PATH ---
            # 1. Multi-scale processing & input prep
            x_enc_scales, _ = self.__multi_scale_process_inputs(x_enc, None)
            x_list = []
            for i, x in enumerate(x_enc_scales):
                B_s, T_s, N_s = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.configs.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B_s * N_s, T_s, 1)
                x_list.append(x)
            
            # 2. TimeMixer feature extraction
            features_list = self.timemixer(x_list, x_mark_list=None)
            
            # 3. Take highest resolution features
            enc_out = features_list[0]
            if self.configs.channel_independence:
                enc_out = enc_out.view(B, N, S, -1)

            features_for_downstream = enc_out

            # 4. Optional News Fusion
            if self.has_news:
                if news_features is None:
                    raise ValueError("Model initialized to use news, but no news_features were provided.")
                processed_news = self.news_processor(news_features)
                news_feat_expand = processed_news.unsqueeze(2).expand(-1, -1, S, -1)
                features_for_downstream = torch.cat([features_for_downstream, news_feat_expand], dim=-1)

            # 5. Optional GCN
            if self.use_gcn:
                if edge_index is None:
                    raise ValueError("Model initialized to use GCN, but no edge_index was provided.")
                gcn_input = features_for_downstream.permute(0, 2, 1, 3).reshape(B * S * N, -1)
                edge_index_batch = torch.cat([edge_index + j * N for j in range(B * S)], dim=1)

                # 处理边权重
                if edge_weight is not None:
                    edge_weight_batch = edge_weight.repeat(B * S)
                    gcn_out = self.gcn_layers[0](gcn_input, edge_index_batch, edge_weight_batch)
                else:
                    gcn_out = self.gcn_layers[0](gcn_input, edge_index_batch)

                features_for_downstream = gcn_out.view(B, S, N, -1).permute(0, 2, 1, 3)

            # 6. Combine with time encoding
            x_mark_enc_expanded = x_mark_enc.unsqueeze(1).expand(-1, N, -1, -1)
            combined_features = torch.cat([features_for_downstream, x_mark_enc_expanded], dim=-1)

            # 7. Classification Head
            output = self.act(combined_features)
            output = self.dropout(output)
            output = output.reshape(B * N, -1)
            output = self.projection(output)
            output = output.view(B, N, -1)
            
            return output

        else:
            # --- REGRESSION PATH ---
            x_enc_scales, x_mark_enc_scales = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
            
            x_list = []
            x_mark_list = []
            if x_mark_enc is not None:
                for i, (x, x_mark) in enumerate(zip(x_enc_scales, x_mark_enc_scales)):
                    B_s, T_s, N_s = x.size()
                    x = self.normalize_layers[i](x, 'norm')
                    if self.configs.channel_independence:
                        x = x.permute(0, 2, 1).contiguous().reshape(B_s * N_s, T_s, 1)
                        x_list.append(x)
                        x_mark = x_mark.repeat(N_s, 1, 1)
                        x_mark_list.append(x_mark)
                    else:
                        x_list.append(x)
                        x_mark_list.append(x_mark)
            else:
                for i, x in enumerate(x_enc_scales):
                    B_s, T_s, N_s = x.size()
                    x = self.normalize_layers[i](x, 'norm')
                    if self.configs.channel_independence:
                        x = x.permute(0, 2, 1).contiguous().reshape(B_s * N_s, T_s, 1)
                    x_list.append(x)
            
            features_list = self.timemixer(x_list, x_mark_list)
            
            processed_news = self.news_processor(news_features) if self.has_news and news_features is not None else None

            dec_out_list = []
            for i, features_scale in enumerate(features_list):
                B_s, T_prime, d_model = features_scale.size()
                
                features_reshaped = features_scale.view(B, N, T_prime, d_model)
                
                features_for_downstream = features_reshaped

                if self.has_news:
                    if processed_news is None:
                        raise ValueError("Model initialized to use news, but no news_features were provided.")
                    news_feat_expand = processed_news.unsqueeze(2).expand(-1, -1, T_prime, -1)
                    features_for_downstream = torch.cat([features_reshaped, news_feat_expand], dim=-1)

                if self.use_gcn:
                    if edge_index is None:
                        raise ValueError("Model initialized to use GCN, but no edge_index was provided.")

                    gcn_input = features_for_downstream.permute(0, 2, 1, 3).reshape(B * T_prime * N, -1)
                    edge_index_batch = torch.cat([edge_index + j * N for j in range(B * T_prime)], dim=1)

                    # 处理边权重
                    if edge_weight is not None:
                        edge_weight_batch = edge_weight.repeat(B * T_prime)
                        gcn_out = self.gcn_layers[i](gcn_input, edge_index_batch, edge_weight_batch)
                    else:
                        gcn_out = self.gcn_layers[i](gcn_input, edge_index_batch)

                    features_for_downstream = gcn_out.view(B, T_prime, N, -1).permute(0, 2, 1, 3)
                
                aggregated_features = features_for_downstream.mean(dim=2)
                aggregated_features_flat = aggregated_features.reshape(B * N, -1)
                
                dec_out = self.predict_layers[i](aggregated_features_flat)
                dec_out = dec_out.view(B, N, -1)
                dec_out_list.append(dec_out)
            
            final_features = torch.stack(dec_out_list, dim=-1).sum(-1)
            
            final_features_flat = final_features.view(-1, final_features.size(-1))
            output_flat = self.mlp(final_features_flat)
            
            output = output_flat.view(B, N, -1)
            if self.task_type == 'regression':
                output = output.squeeze(-1)
            return output 