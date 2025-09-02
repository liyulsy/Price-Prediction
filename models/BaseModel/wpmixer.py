#!/usr/bin/env python3
"""
WPMixer (Wavelet Patch Mixer) 基础模型
从原始WPMixer项目搬移的核心实现

主要功能：
1. 小波分解和重构
2. 补丁嵌入
3. MLP混合
4. 多分辨率分支处理

作者：基于WPMixer项目搬移
"""

import torch
import torch.nn as nn

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.RevIN import RevIN
from models.layers.decomposition import Decomposition
from models.layers.wpmixer_layers import ResolutionBranch

class WPMixerCore(nn.Module):
    """WPMixer核心模型 - 从原始项目搬移，移除最后的预测层"""

    def __init__(self,
                 input_length = None,
                 pred_length = None,
                 wavelet_name = None,
                 level = None,
                 batch_size = None,
                 channel = None,
                 d_model = None,
                 dropout = None,
                 embedding_dropout = None,
                 tfactor = None,
                 dfactor = None,
                 device = None,
                 patch_len = None,
                 patch_stride = None,
                 no_decomposition = None,
                 use_amp = None):

        super(WPMixerCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp

        self.Decomposition_model = Decomposition(input_length = self.input_length,
                                        pred_length = self.pred_length,
                                        wavelet_name = self.wavelet_name,
                                        level = self.level,
                                        batch_size = self.batch_size,
                                        channel = self.channel,
                                        d_model = self.d_model,
                                        tfactor = self.tfactor,
                                        dfactor = self.dfactor,
                                        device = self.device,
                                        no_decomposition = self.no_decomposition,
                                        use_amp = self.use_amp)

        self.input_w_dim = self.Decomposition_model.input_w_dim # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.input_w_dim[i],
                                                           pred_seq = self.pred_w_dim[i],
                                                           batch_size = self.batch_size,
                                                           channel = self.channel,
                                                           d_model = self.d_model,
                                                           dropout = self.dropout,
                                                           embedding_dropout = self.embedding_dropout,
                                                           tfactor = self.tfactor,
                                                           dfactor = self.dfactor,
                                                           patch_len = self.patch_len,
                                                           patch_stride = self.patch_stride) for i in range(len(self.input_w_dim))])

        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)

    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        features : 提取的特征表示，移除了最后的预测层
        '''

        # RevIN 期望输入格式: [batch, seq_len, num_features]
        # 但 xL 的格式是: [batch, num_features, seq_len]
        xL_for_revin = xL.transpose(1, 2)  # [batch, seq_len, num_features]
        x = self.revin(xL_for_revin, 'norm')  # [batch, seq_len, num_features]
        x = x.transpose(1, 2) # [batch, num_features, seq_len]

        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        # 执行逆小波变换重构预测序列
        reconstructed_x = self.Decomposition_model.inv_transform(yA, yD)

        # 执行反归一化 (RevIN denormalization)
        reconstructed_x = reconstructed_x.transpose(1, 2) # -> [B, L_pred, C]
        reconstructed_x = reconstructed_x[:, -self.pred_length:, :]  # 先裁剪时间维度
        denorm_x = self.revin(reconstructed_x, 'denorm')
        denorm_x = denorm_x.transpose(1, 2) # -> [B, C, L_pred]

        # 返回正确裁剪的结果
        return denorm_x
    
        #     # 对于特征提取，我们需要更丰富的表示
        # # 不进行逆变换，而是直接使用分解后的特征

        # # 将所有分辨率的特征连接起来
        # all_features = [yA]  # 低频特征
        # all_features.extend(yD)  # 高频特征

        # # 将所有特征在最后一个维度上连接
        # # yA: [batch, channel, pred_length], yD: [batch, channel, pred_length]
        # concatenated_features = torch.cat(all_features, dim=-1)  # [batch, channel, total_features]

        # # 转换为期望的格式: [batch, channel, d_model]
        # # 使用平均池化或线性变换来获得固定的d_model维度
        # if concatenated_features.size(-1) != self.d_model:
        #     # 使用自适应平均池化来调整最后一个维度
        #     concatenated_features = F.adaptive_avg_pool1d(
        #         concatenated_features, self.d_model
        #     )  # [batch, channel, d_model]

        # return concatenated_features  # [batch, channel, d_model]


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试WPMixer核心模型...")

    # 创建模型参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WPMixerCore(
        input_length=96,
        pred_length=24,
        wavelet_name='db4',
        level=3,
        batch_size=32,
        channel=8,
        d_model=128,
        dropout=0.1,
        embedding_dropout=0.1,
        tfactor=2,
        dfactor=4,
        device=device,
        patch_len=16,
        patch_stride=8,
        no_decomposition=False,
        use_amp=False
    )

    # 创建测试数据
    batch_size = 32
    x = torch.randn(batch_size, 96, 8)

    # 前向传播
    try:
        output = model(x)
        print(f"✅ 输入形状: {x.shape}")
        print(f"✅ 输出形状: {output.shape}")
        print(f"✅ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print("🎉 WPMixer核心模型测试通过！")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("💡 这可能是因为缺少pytorch_wavelets库，模型会使用简化的分解方法")
