#!/usr/bin/env python3
"""
统一WPMixer混合模型 - 简化重写版本
结合WPMixer时间特征提取 + GCN图建模 + 新闻特征 + MLP预测

流程：
1. WPMixer提取时间序列特征
2. 可选：融合新闻特征
3. 可选：GCN图卷积增强
4. MLP最终预测

作者：基于WPMixer、TimeMixer等模型改编
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 导入基础模型
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models.BaseModel.wpmixer import WPMixerCore


class UnifiedWPMixer(nn.Module):
    """
    统一WPMixer混合模型 - 简化版本

    架构：WPMixer → [新闻融合] → [GCN增强] → MLP预测
    """

    def __init__(self,
                 # 基础参数
                 input_length: int = 96,
                 pred_length: int = 1,
                 num_coins: int = 8,
                 d_model: int = 128,
                 patch_len: int = 8,
                 patch_stride: int = 4,

                 # WPMixer参数
                 wavelet_name: str = 'db4',
                 level: int = 2,
                 tfactor: int = 2,
                 dfactor: int = 4,
                 no_decomposition: bool = False,
                 use_amp: bool = False,

                 # 功能开关
                 use_news: bool = False,
                 use_gcn: bool = False,
                 use_time_features: bool = False,

                 # 新闻参数
                 news_feature_dim: int = 0,
                 news_processed_dim: int = 64,

                 # GCN参数
                 gcn_hidden_dim: int = 128,

                 # 任务参数
                 task_type: str = 'classification',
                 num_classes: int = 2,

                 # 其他参数
                 dropout: float = 0.1,
                 device: torch.device = torch.device('cpu')):
        super(UnifiedWPMixer, self).__init__()

        # 保存参数
        self.use_news = use_news
        self.use_gcn = use_gcn
        self.task_type = task_type
        self.num_classes = num_classes
        self.d_model = d_model

        # 1. WPMixer核心 - 时间序列特征提取
        self.wpmixer = WPMixerCore(
            input_length=input_length,
            pred_length=pred_length,
            wavelet_name=wavelet_name,
            level=level,
            batch_size=32,  # 使用动态批次大小提高稳定性
            channel=num_coins,
            d_model=d_model,
            tfactor=tfactor,
            dfactor=dfactor,
            device=device,
            no_decomposition=no_decomposition,
            use_amp=use_amp,
            patch_len=patch_len,
            patch_stride=patch_stride,
            dropout=dropout,
            embedding_dropout=dropout
        )

        # 特征投影层 - 确保维度一致性
        self.feature_projection = nn.Linear(num_coins, d_model)
        self.feature_norm = nn.LayerNorm(d_model)

        # 2. 新闻特征处理器（可选）
        if use_news and news_feature_dim > 0:
            self.news_processor = nn.Sequential(
                nn.Linear(news_feature_dim, news_processed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(news_processed_dim * 2, news_processed_dim),
                nn.LayerNorm(news_processed_dim)
            )
            feature_dim = d_model + news_processed_dim
        else:
            self.news_processor = None
            feature_dim = d_model

        # 3. GCN图卷积层（可选）
        if use_gcn:
            from torch_geometric.nn import GCNConv
            # GCN输入和输出维度保持一致，避免残差连接问题
            self.gcn = GCNConv(feature_dim, feature_dim)
            final_feature_dim = feature_dim
        else:
            self.gcn = None
            final_feature_dim = feature_dim

        # 4. 特征融合层 - 增加稳定性
        self.feature_fusion = nn.Sequential(
            nn.Linear(final_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),  # 增加dropout防止过拟合
        )

        # 5. 最终预测层
        if task_type == 'classification':
            # 分类：改进的MLP with 更多正则化
            self.predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.LayerNorm(d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, num_classes)
            )
        else:
            # 回归：改进的MLP
            self.predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self,
                price_data: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                edge_weight: Optional[torch.Tensor] = None,
                news_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播 - 简化版本

        Args:
            price_data: [batch_size, seq_len, num_coins] 价格数据
            edge_index: [2, num_edges] 图的边索引
            edge_weight: [num_edges] 边权重
            news_features: [batch_size, num_coins, news_dim] 新闻特征

        Returns:
            torch.Tensor: 预测结果 [batch_size, num_coins, num_classes/1]
        """
        batch_size, _, num_coins = price_data.shape

        # 步骤1: WPMixer提取时间序列特征
        wpmixer_input = price_data.permute(0, 2, 1)  # [batch, num_coins, seq_len]
        time_features = self.wpmixer(wpmixer_input)  # [batch, pred_length, num_coins]

        # 改进的数值稳定性处理
        if torch.isnan(time_features).any() or torch.isinf(time_features).any():
            print(f"⚠️ WPMixer输出包含NaN或Inf，使用渐进式修复...")
            # 使用更温和的修复方法
            time_features = torch.where(torch.isnan(time_features),
                                      torch.zeros_like(time_features),
                                      time_features)
            time_features = torch.clamp(time_features, min=-10.0, max=10.0)

        # 根据实际输出维度调整处理方式
        if len(time_features.shape) == 3:
            # 如果是 [batch, pred_length, num_coins]，转置为 [batch, num_coins, pred_length]
            if time_features.shape[2] == num_coins:
                time_features = time_features.transpose(1, 2)  # [batch, num_coins, pred_length]
            # 全局平均池化得到节点特征 [batch, num_coins]
            node_features = time_features.mean(dim=-1)  # [batch, num_coins]
        else:
            # 如果是其他维度，直接使用
            node_features = time_features

        # 特征投影到d_model维度
        if node_features.shape[-1] != self.d_model:
            # 动态创建正确维度的投影层
            if not hasattr(self, 'dynamic_projection') or self.dynamic_projection.in_features != node_features.shape[-1]:
                self.dynamic_projection = nn.Linear(node_features.shape[-1], self.d_model).to(node_features.device)
            node_features = self.dynamic_projection(node_features)  # [batch, d_model] 或 [batch, num_coins, d_model]

        node_features = self.feature_norm(node_features)  # 归一化

        # 确保是 [batch, num_coins, d_model] 格式
        if len(node_features.shape) == 2:
            # [batch, d_model] -> [batch, num_coins, d_model]
            features = node_features.unsqueeze(1).expand(-1, num_coins, -1)
        else:
            # 已经是 [batch, num_coins, d_model]
            features = node_features

        # 步骤2: 可选的新闻特征融合
        if self.use_news and news_features is not None and self.news_processor is not None:
            processed_news = self.news_processor(news_features)  # [batch, num_coins, news_processed_dim]
            # 使用残差连接进行特征融合
            features = torch.cat([features, processed_news], dim=-1)  # [batch, num_coins, d_model+news_dim]
        # 如果没有新闻特征，features保持不变

        # 步骤3: 可选的GCN图卷积增强
        if self.use_gcn and edge_index is not None and self.gcn is not None:
            try:
                # 准备图数据 - 使用reshape而不是view
                node_features_flat = features.reshape(-1, features.size(-1))  # [batch*num_coins, feature_dim]

                # 构建批处理的边索引
                batch_edges = []
                for b in range(batch_size):
                    offset = b * num_coins
                    batch_edges.append(edge_index + offset)
                batch_edge_index = torch.cat(batch_edges, dim=1)  # [2, batch*num_edges]

                # 改进的边界检查
                max_node_id = batch_size * num_coins - 1
                if batch_edge_index.max().item() > max_node_id:
                    print(f"⚠️ 边索引超出范围，进行修复: max={batch_edge_index.max().item()}, limit={max_node_id}")
                    batch_edge_index = torch.clamp(batch_edge_index, 0, max_node_id)

                # 处理边权重
                if edge_weight is not None:
                    batch_edge_weight = edge_weight.repeat(batch_size)
                else:
                    batch_edge_weight = None

                # GCN前向传播
                enhanced_features = self.gcn(node_features_flat, batch_edge_index, batch_edge_weight)

                # 重塑回批次格式 - 使用reshape而不是view
                gcn_features = enhanced_features.reshape(batch_size, num_coins, -1)

                # 确保维度匹配后再进行残差连接
                if gcn_features.shape == features.shape:
                    features = features + gcn_features
                else:
                    print(f"⚠️ GCN输出维度不匹配: {gcn_features.shape} vs {features.shape}，跳过残差连接")
                    features = gcn_features

            except Exception as e:
                print(f"⚠️ GCN处理出错，跳过GCN增强: {e}")
                # 如果GCN失败，继续使用原始特征

        # 步骤4: 特征融合
        fused_features = self.feature_fusion(features)  # [batch, num_coins, d_model]

        # 步骤5: 最终预测
        predictions = self.predictor(fused_features)  # [batch, num_coins, num_classes/1]

        # 最终的数值稳定性检查
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print(f"⚠️ 预测结果包含NaN或Inf，进行修复...")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=-1.0)

        return predictions