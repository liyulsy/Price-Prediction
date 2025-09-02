import torch
import torch.nn as nn
from ..BaseModel.wpmixer import WPMixerCore
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

class UnifiedWPMixer(nn.Module):
    def __init__(self,
                 configs,
                 gcn_hidden_dim,
                 gcn_output_dim,
                 # --- Optional Components ---
                 use_gcn: bool = False,
                 gcn_config: str = 'improved_gelu',  # 新增：GCN配置选择
                 news_feature_dim: int = None,
                 news_processed_dim: int = 32,
                 # --- MLP and Output ---
                 mlp_hidden_dim: int = 256,
                 num_classes: int = 2):
        super(UnifiedWPMixer, self).__init__()

        self.task_type = configs.task_type
        self.use_gcn = use_gcn
        self.gcn_config = gcn_config
        self.has_news = news_feature_dim is not None and news_feature_dim > 0
        self.num_classes = num_classes


        # 1. WPMixer核心 - 时间序列特征提取
        self.wpmixer = WPMixerCore(
            input_length=configs.input_length,
            pred_length=configs.pred_length,
            wavelet_name=configs.wavelet_name,
            level=configs.level,
            batch_size=32,  # 使用动态批次大小提高稳定性
            channel=1,  # 单币种独立处理
            d_model=configs.d_model,
            dropout=configs.dropout,
            embedding_dropout=configs.dropout,
            tfactor=configs.tfactor,
            dfactor=configs.dfactor,
            patch_len=configs.patch_len,
            patch_stride=configs.patch_stride,
            no_decomposition=configs.no_decomposition,
            use_amp=configs.use_amp
        )

        # 计算基础特征维度（WPMixer输出维度）
        # 直接使用pred_length作为特征维度，不投影到d_model
        base_features_dim = configs.pred_length

        # --- Optional News Processor ---
        if self.has_news:
            self.news_processor = MLPBlock(
                input_dim=news_feature_dim,
                hidden_dim=news_processed_dim * 2,
                output_dim=news_processed_dim,
                dropout=configs.dropout if hasattr(configs, 'dropout') else 0.3
            )
            # 计算融合后的特征维度
            features_dim = base_features_dim + news_processed_dim
        else:
            self.news_processor = None
            features_dim = base_features_dim

        # --- Optional GCN ---
        if self.use_gcn:
            # 使用改进的GCN配置
            print(f"🔧 UnifiedWPMixer使用GCN配置: {self.gcn_config}")
            self.gcn = create_gcn_from_config(self.gcn_config, features_dim, gcn_hidden_dim, gcn_output_dim)
            mlp_input_dim = gcn_output_dim  # MLP takes output from GCN
        else:
            self.gcn = None
            mlp_input_dim = features_dim  # MLP takes features directly


        # 5. 最终预测层
        if self.task_type == 'regression':
            final_layer_output_dim = 1 # Predict one continuous value
        else: # 'classification'
            final_layer_output_dim = num_classes

        self.mlp = MLPBlock(
            input_dim=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=final_layer_output_dim,
            dropout=configs.dropout if hasattr(configs, 'dropout') else 0.3
        )


    def forward(self, price_data, edge_index=None, edge_weight=None, news_features=None):
        """
        前向传播

        Args:
            price_data: [batch_size, seq_len, num_coins] 价格序列数据
            edge_index: [2, num_edges] GCN边索引（可选）
            edge_weight: [num_edges] GCN边权重（可选）
            news_features: [batch_size, num_coins, news_dim] 新闻特征（可选）

        Returns:
            predictions: [batch_size, num_coins, num_classes/1] 预测结果
        """
        batch_size, seq_len, num_coins = price_data.shape

        # 1. WPMixer时间序列特征提取 - 优化的逐币种处理
        # 转置为 [batch_size, num_coins, seq_len]
        price_transposed = price_data.transpose(1, 2)  # [batch_size, num_coins, seq_len]

        # 使用列表推导式和torch.stack进行高效处理
        wpmixer_outputs = []
        for coin_idx in range(num_coins):
            # WPMixer输出: [batch_size, 1, pred_length]
            wpmixer_out = self.wpmixer(price_transposed[:, coin_idx, :].unsqueeze(1))
            # 去掉通道维度，保留 pred_length 作为特征维度
            coin_features = wpmixer_out.squeeze(1)  # [batch_size, pred_length]
            wpmixer_outputs.append(coin_features)

        # 合并所有币种的特征 [batch_size, num_coins]
        wpmixer_features = torch.stack(wpmixer_outputs, dim=1)  # [batch_size, num_coins, pred_length]

        # # # 扩展特征维度以匹配后续处理
        # if len(wpmixer_features.shape) == 2:
        #     wpmixer_features = wpmixer_features.unsqueeze(-1)  # [batch_size, num_coins, 1]

        # 2. 融合新闻特征（可选）
        if self.has_news and self.news_processor is not None and news_features is not None:
            # 处理新闻特征 [batch_size, num_coins, news_dim] -> [batch_size, num_coins, news_processed_dim]
            news_processed = self.news_processor(news_features)
            # 拼接特征 [batch_size, num_coins, d_model + news_processed_dim]
            combined_features = torch.cat([wpmixer_features, news_processed], dim=-1)
        else:
            combined_features = wpmixer_features

        # 3. GCN图卷积增强（可选）
        if self.use_gcn and self.gcn is not None and edge_index is not None:
            # 重塑为图数据格式 [batch_size * num_coins, feature_dim]
            graph_features = combined_features.reshape(batch_size * num_coins, -1)

            # 扩展边索引以处理批次数据
            batch_edge_index = []
            batch_edge_weight = []
            for b in range(batch_size):
                offset = b * num_coins
                batch_edge_index.append(edge_index + offset)
                if edge_weight is not None:
                    batch_edge_weight.append(edge_weight)

            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            if edge_weight is not None:
                batch_edge_weight = torch.cat(batch_edge_weight, dim=0)
            else:
                batch_edge_weight = None

            # GCN处理
            gcn_features = self.gcn(graph_features, batch_edge_index, batch_edge_weight)

            # 重塑回 [batch_size, num_coins, gcn_output_dim]
            final_features = gcn_features.reshape(batch_size, num_coins, -1)
        else:
            final_features = combined_features

        # 4. 最终预测 - 使用MLP
        B, N, F = final_features.shape
        mlp_in = final_features.view(B * N, F)
        mlp_out = self.mlp(mlp_in)
        predictions = mlp_out.view(B, N, -1)

        # 5. 根据任务类型调整输出形状
        if self.task_type == 'regression':
            # 回归任务：预测每个币种的价格，输出 [batch_size, num_coins]
            predictions = predictions.squeeze(-1)  # 移除最后一个维度
        else:
            # 分类任务：预测每个币种的类别，输出 [batch_size, num_coins, num_classes]
            pass  # 保持原有形状

        return predictions


















