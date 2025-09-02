#!/usr/bin/env python3
"""
混合重构WPMixer模型

架构: 特征提取/融合 (类似UnifiedWPMixer) + 重构头 (类似原始WPMixer)
目的: 测试移除逆小波变换是否是导致R²为负的原因。
"""

import torch
import torch.nn as nn

from ..BaseModel.wpmixer import WPMixerCore
from models.BaseModel.improved_gcn import create_gcn_from_config
from .unified_wpmixer import MLPBlock

class HybridReconstructionWPMixer(nn.Module):
    def __init__(self, configs, gcn_hidden_dim, gcn_output_dim, use_gcn=False, gcn_config='improved_gelu',
                 news_feature_dim=None, news_processed_dim=32, batch_size=32, task_type='regression', num_classes=2):
        super().__init__()
        self.use_gcn = use_gcn
        self.has_news = news_feature_dim is not None and news_feature_dim > 0
        self.configs = configs
        self.task_type = task_type
        self.num_classes = num_classes

        # 1. WPMixer核心 - 现在是完整的预测模型
        self.wpmixer = WPMixerCore(
            input_length=configs.input_length,
            pred_length=configs.pred_length,
            wavelet_name=configs.wavelet_name,
            level=configs.level,
            batch_size=batch_size,
            channel=1, # WPMixerCore 必须在单通道模式下工作
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

        # --- 增强模块 ---
        # 基础特征维度现在是预测序列的长度
        base_features_dim = configs.pred_length

        # 2. 新闻处理器 (如果使用)
        if self.has_news:
            self.news_processor = MLPBlock(
                input_dim=news_feature_dim,
                hidden_dim=news_processed_dim * 2,
                output_dim=news_processed_dim
            )
            features_dim = base_features_dim + news_processed_dim
        else:
            self.news_processor = None
            features_dim = base_features_dim

        # 3. GCN (如果使用)
        if self.use_gcn:
            self.gcn = create_gcn_from_config(gcn_config, features_dim, gcn_hidden_dim, gcn_output_dim)
            final_mlp_input_dim = gcn_output_dim
        else:
            self.gcn = None
            final_mlp_input_dim = features_dim

        # 4. 最终预测头
        # 4. 最终预测头
        if self.task_type == 'classification':
            final_output_dim = self.num_classes
        else:  # regression
            final_output_dim = configs.pred_length

        self.final_mlp = MLPBlock(
            input_dim=final_mlp_input_dim,
            hidden_dim=configs.d_model,
            output_dim=final_output_dim
        )

    def forward(self, price_data, edge_index=None, edge_weight=None, news_features=None):
        batch_size, _, num_coins = price_data.shape
        price_transposed = price_data.transpose(1, 2)  # -> [B, num_coins, seq_len]

        # 1. 为每个币种生成初步预测
        wpmixer_preds_list = [
            self.wpmixer(price_transposed[:, i, :].unsqueeze(1)) for i in range(num_coins)
        ]
        wpmixer_preds = torch.cat(wpmixer_preds_list, dim=1) # -> [B, num_coins, pred_len]

        # 2. 特征融合 (初步预测 + 新闻)
        combined_features = wpmixer_preds
        if self.has_news and self.news_processor is not None and news_features is not None:
            news_processed = self.news_processor(news_features)
            combined_features = torch.cat([wpmixer_preds, news_processed], dim=-1)

        # 3. GCN处理
        if self.use_gcn and self.gcn is not None and edge_index is not None:
            graph_features = combined_features.reshape(batch_size * num_coins, -1)

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

            gcn_features = self.gcn(graph_features, batch_edge_index, batch_edge_weight)
            # The output shape of gcn_features is [B * num_coins, gcn_output_dim]
            final_features = gcn_features.view(batch_size, num_coins, -1)
        else:
            final_features = combined_features

        # 4. 最终预测
        final_features_reshaped = final_features.reshape(batch_size * num_coins, -1)
        predictions_reshaped = self.final_mlp(final_features_reshaped)
        if self.task_type == 'classification':
            predictions = predictions_reshaped.view(batch_size, num_coins, self.num_classes)
        else:
            predictions = predictions_reshaped.view(batch_size, num_coins, self.configs.pred_length)

        return predictions


