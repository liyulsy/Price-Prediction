#!/usr/bin/env python3
"""
时间特征提取模块
从WPMixer项目中剥离出来的时间特征处理组件

主要功能：
1. 提取各种时间特征（小时、天、月等）
2. 时间编码
3. 周期性特征处理

作者：基于WPMixer项目改编
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from datetime import datetime


class TimeFeature:
    """时间特征基类"""
    
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """分钟中的秒数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """小时中的分钟数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """一天中的小时数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """一周中的天数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """一月中的天数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """一年中的天数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """一年中的月份，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """一年中的周数，编码为[-0.5, 0.5]之间的值"""
    
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    根据频率字符串返回相应的时间特征列表
    
    Args:
        freq_str: 频率字符串，如 'h', 'd', 'M' 等
        
    Returns:
        List[TimeFeature]: 时间特征列表
    """
    features_by_offsets = {
        'S': [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        'T': [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        'H': [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        'D': [DayOfWeek, DayOfMonth, DayOfYear],
        'B': [DayOfWeek, DayOfMonth, DayOfYear],
        'W': [DayOfMonth, WeekOfYear],
        'M': [MonthOfYear],
        'Q': [MonthOfYear],
        'Y': [],
    }
    
    # 标准化频率字符串
    freq_str = freq_str.upper()
    if freq_str in features_by_offsets:
        return [cls() for cls in features_by_offsets[freq_str]]
    else:
        # 默认使用小时级特征
        return [cls() for cls in features_by_offsets['H']]


class TimeFeatureEmbedding(nn.Module):
    """时间特征嵌入模块"""
    
    def __init__(self, 
                 d_model: int = 128,
                 freq: str = 'h',
                 embed_type: str = 'timeF'):
        super(TimeFeatureEmbedding, self).__init__()
        
        self.d_model = d_model
        self.freq = freq
        self.embed_type = embed_type
        
        # 获取时间特征
        self.time_features = time_features_from_frequency_str(freq)
        self.num_time_features = len(self.time_features)
        
        if embed_type == 'timeF':
            # 直接使用时间特征
            self.embed_layer = nn.Linear(self.num_time_features, d_model)
        elif embed_type == 'fixed':
            # 固定位置编码
            self.embed_layer = nn.Embedding(1000, d_model)  # 假设最大1000个时间步
        elif embed_type == 'learned':
            # 可学习的位置编码
            self.embed_layer = nn.Parameter(torch.randn(1000, d_model))
        else:
            raise ValueError(f"不支持的嵌入类型: {embed_type}")
    
    def forward(self, x: torch.Tensor, time_index: Optional[pd.DatetimeIndex] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, features]
            time_index: 时间索引
            
        Returns:
            torch.Tensor: 时间特征嵌入 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        if self.embed_type == 'timeF' and time_index is not None:
            # 提取时间特征
            time_features = []
            for feature_func in self.time_features:
                feature_values = feature_func(time_index)
                time_features.append(feature_values)
            
            # 堆叠时间特征
            time_features = np.stack(time_features, axis=-1)  # [seq_len, num_features]
            time_features = torch.from_numpy(time_features).float().to(x.device)
            
            # 扩展到batch维度
            time_features = time_features.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 嵌入
            time_embed = self.embed_layer(time_features)
            
        elif self.embed_type == 'fixed':
            # 固定位置编码
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            time_embed = self.embed_layer(positions)
            
        elif self.embed_type == 'learned':
            # 可学习位置编码
            time_embed = self.embed_layer[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            
        else:
            # 默认零嵌入
            time_embed = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        
        return time_embed


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的张量
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


class CyclicalTimeEncoding(nn.Module):
    """周期性时间编码"""
    
    def __init__(self, d_model: int):
        super(CyclicalTimeEncoding, self).__init__()
        self.d_model = d_model
        
        # 不同周期的编码
        self.hour_embed = nn.Embedding(24, d_model // 4)
        self.day_embed = nn.Embedding(7, d_model // 4)
        self.month_embed = nn.Embedding(12, d_model // 4)
        self.year_embed = nn.Embedding(10, d_model // 4)  # 假设10年范围
        
    def forward(self, time_index: pd.DatetimeIndex) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time_index: 时间索引
            
        Returns:
            torch.Tensor: 周期性时间编码 [seq_len, d_model]
        """
        device = next(self.parameters()).device
        
        # 提取时间组件
        hours = torch.tensor(time_index.hour, device=device)
        days = torch.tensor(time_index.dayofweek, device=device)
        months = torch.tensor(time_index.month - 1, device=device)  # 0-11
        years = torch.tensor((time_index.year - time_index.year.min()) % 10, device=device)
        
        # 嵌入
        hour_emb = self.hour_embed(hours)
        day_emb = self.day_embed(days)
        month_emb = self.month_embed(months)
        year_emb = self.year_embed(years)
        
        # 拼接
        time_encoding = torch.cat([hour_emb, day_emb, month_emb, year_emb], dim=-1)
        
        return time_encoding


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试时间特征提取模块...")
    
    # 创建测试时间索引
    dates = pd.date_range('2023-01-01', periods=96, freq='H')
    
    # 测试时间特征
    hour_feature = HourOfDay()
    hour_values = hour_feature(dates)
    print(f"✅ 小时特征形状: {hour_values.shape}")
    print(f"✅ 小时特征范围: [{hour_values.min():.3f}, {hour_values.max():.3f}]")
    
    # 测试时间特征嵌入
    time_embed = TimeFeatureEmbedding(d_model=128, freq='h')
    x = torch.randn(32, 96, 8)
    time_features = time_embed(x, dates)
    print(f"✅ 时间嵌入形状: {time_features.shape}")
    
    # 测试位置编码
    pos_encoding = PositionalEncoding(d_model=128)
    pos_features = pos_encoding(torch.randn(32, 96, 128))
    print(f"✅ 位置编码形状: {pos_features.shape}")
    
    print("🎉 时间特征提取模块测试通过！")
