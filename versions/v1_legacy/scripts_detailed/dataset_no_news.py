import torch
import pandas as pd
from torch.utils.data import Dataset
import os # 保留 os 用于可能的路径操作，即使新闻缓存没了
import numpy as np
# load_news_data 函数可以保留在此文件中作为通用工具，即使 CryptoDatasetNoNews 不使用它
# 或者根据需要决定是否也将其移动或删除。为保持与原文件结构的某种相似性，暂时保留。
import json

def time_features(dates, freq='h'):
    """
    Extract time features from a pd.DatetimeIndex.
    `dates` should be a pd.DatetimeIndex.
    `freq` indicates the frequency of the data.
    Returns a numpy array of shape (len(dates), num_features).
    """
    if isinstance(dates, np.ndarray) and np.issubdtype(dates.dtype, np.datetime64):
        dates = pd.to_datetime(dates) 
    elif not isinstance(dates, pd.DatetimeIndex):
        # If a single Timestamp is passed, convert to DatetimeIndex
        if isinstance(dates, pd.Timestamp):
            dates = pd.DatetimeIndex([dates])
        else:
            raise TypeError("Input `dates` must be a pd.DatetimeIndex, pd.Timestamp, or numpy array of datetime64.")

    features = []
    features.append(dates.month / 12.0 - 0.5)
    features.append(dates.day / 31.0 - 0.5)
    features.append(dates.dayofweek / 6.0 - 0.5) 
    features.append(dates.dayofyear / 365.0 - 0.5) 
    features.append(dates.hour / 23.0 - 0.5) 
    features.append(dates.isocalendar().week.values.astype(float) / 52.0 - 0.5) # Ensure week is float for tensor
    
    return np.stack(features, axis=1).astype(np.float32) # Ensure float32 for torch tensor

def load_news_data(features_dir, coin_names):
    """加载所有币种的新闻数据（此函数在此文件中可能不被 CryptoDatasetNoNews 使用）"""
    news_data_dict = {}
    for coin in coin_names:
        file_path = os.path.join(features_dir, f"{coin.replace(' ', '')}_features.json")
        try:
            with open(file_path, 'r') as f:
                news_data_dict[coin] = json.load(f)
        except FileNotFoundError:
            # print(f"警告: 未找到 {coin} 的新闻文件: {file_path}")
            news_data_dict[coin] = []
        except json.JSONDecodeError:
            # print(f"警告: 解析 {coin} 的新闻文件失败: {file_path}")
            news_data_dict[coin] = []
    return news_data_dict

class CryptoDatasetNoNews(Dataset):
    def __init__(self, 
                 price_data_df: pd.DataFrame, 
                 seq_len: int = 24,
                 time_encoding_enabled: bool = False, # 添加时间编码开关参数
                 time_freq: str = 'h'): # 添加时间频率参数
        """
        Args:
            price_data_df: DataFrame, 包含所有币种的价格数据，索引为时间戳，列为币种名称。
            seq_len: 序列长度，模型输入的时间步数。
            time_encoding_enabled: 是否启用时间特征编码。
            time_freq: 时间特征的频率（例如 'h'）。
        """
        super().__init__()
        self.price_data_df = price_data_df
        self.seq_len = seq_len
        self.time_index = pd.to_datetime(self.price_data_df.index) # 确保时间索引是 datetime 类型

        # 从传入的price_data_df获取币种列表和顺序
        self.coin_names = list(price_data_df.columns) 
        self.coin_to_idx = {name: i for i, name in enumerate(self.coin_names)}
        self.num_coins = len(self.coin_names)
        
        # 无新闻特征，所以维度为0
        # self.news_feature_dim = 0 # 移除新闻特征相关的变量和注释
        # processed_news_features 是一个形状正确的空张量
        # self.processed_news_features = torch.zeros(len(self.time_index), self.num_coins, 0, dtype=torch.float32) # 移除新闻特征相关的变量和注释

        # 所有与新闻处理、加载、缓存、归一化相关的逻辑均已移除
        self.time_encoding_enabled = time_encoding_enabled
        self.time_freq = time_freq
        # Precompute encoded time stamps
        if self.time_encoding_enabled:
            encoded_stamps_np = time_features(self.time_index, freq=self.time_freq)
            self.all_time_stamps_encoded = torch.tensor(encoded_stamps_np, dtype=torch.float32)
        else:
            # Determine num_time_features for zero tensor
            _dummy_dates = pd.DatetimeIndex([pd.Timestamp('2000-01-01')]) # Create a minimal DatetimeIndex
            num_time_features_dim = time_features(_dummy_dates, freq=self.time_freq).shape[1]
            self.all_time_stamps_encoded = torch.zeros(len(self.time_index), num_time_features_dim, dtype=torch.float32)
        self.num_actual_time_features = self.all_time_stamps_encoded.shape[1]
        print(f"时间特征编码已处理。启用: {self.time_encoding_enabled}, 特征维度: {self.num_actual_time_features}")

    def __len__(self):
        return len(self.time_index) - self.seq_len + 1

    def __getitem__(self, idx):
        end_idx = idx + self.seq_len
        # 获取价格序列
        price_seq_df = self.price_data_df[self.coin_names].iloc[idx:end_idx]
        price_seq_tensor = torch.tensor(price_seq_df.values, dtype=torch.float32)
        
        # 获取目标价格
        target_price_tensor = torch.zeros(self.num_coins, dtype=torch.float32) # 默认目标
        if end_idx < len(self.price_data_df):
            target_price_series = self.price_data_df[self.coin_names].iloc[end_idx]
            target_price_tensor = torch.tensor(target_price_series.values, dtype=torch.float32)

        price_seq_mark = self.all_time_stamps_encoded[idx : idx + self.seq_len]

        # 返回的字典不包含 news_features 键
        return_dict = {
            'price_seq': price_seq_tensor,
            'price_seq_mark': price_seq_mark,
            'target_price': target_price_tensor
        }
        return return_dict

if __name__ == '__main__':
    print("--- 测试 CryptoDatasetNoNews (含时间编码) ---") # 更新测试标题
    # 定义参数
    COIN_NAMES_TEST = ['BTC', 'ETH', 'BNB'] # 使用price_data_df的列名
    SEQ_LEN_TEST = 180
    TIME_ENCODING_TEST = True # 测试时启用时间编码
    TIME_FREQ_TEST = 'h' # 测试时使用小时频率
    
    # 准备模拟价格数据
    date_rng_test = pd.date_range(start='2023-01-01', periods=200, freq='H')
    simulated_price_data_test = pd.DataFrame(
        torch.randn(len(date_rng_test), len(COIN_NAMES_TEST)).numpy(),
        index=date_rng_test,
        columns=COIN_NAMES_TEST 
    )
    print("模拟价格数据 (前5行):")
    print(simulated_price_data_test.head())

    # 创建 CryptoDatasetNoNews 实例
    dataset_no_news = CryptoDatasetNoNews(
        price_data_df=simulated_price_data_test,
        seq_len=SEQ_LEN_TEST,
        time_encoding_enabled=TIME_ENCODING_TEST, # 传递时间编码参数
        time_freq=TIME_FREQ_TEST # 传递时间频率参数
    )

    print(f"\nCryptoDatasetNoNews 实例创建成功。") # 更新打印信息
    print(f"  数据集大小: {len(dataset_no_news)}")
    # print(f"  新闻特征维度 (应为0): {dataset_no_news.news_feature_dim}") # 移除新闻特征维度检查
    print(f"  实际时间特征数量: {dataset_no_news.num_actual_time_features}") # 添加时间特征数量打印
    print(f"  币种列表: {dataset_no_news.coin_names}")

    if len(dataset_no_news) > 0:
        sample = dataset_no_news[0]
        print("\n获取的第一个样本:")
        print(f"  价格序列 (price_seq) shape: {sample['price_seq'].shape}")
        assert sample['price_seq'].shape == (SEQ_LEN_TEST, len(COIN_NAMES_TEST))
        print(f"  目标价格 (target_price) shape: {sample['target_price'].shape}")
        assert sample['target_price'].shape == (len(COIN_NAMES_TEST),)
        
        # 检查时间特征
        if TIME_ENCODING_TEST:
            print(f"  时间特征 (price_seq_mark) shape: {sample['price_seq_mark'].shape}")
            assert 'price_seq_mark' in sample # 断言样本包含时间特征键
            assert sample['price_seq_mark'].shape == (SEQ_LEN_TEST, dataset_no_news.num_actual_time_features) # 断言时间特征形状
        else:
            assert 'price_seq_mark' not in sample # 断言样本不包含时间特征键
            print("  时间特征未启用，样本中不包含 'price_seq_mark' 键 (符合预期)。")

        # if 'news_features' in sample: # 移除新闻特征检查
        #     print("错误: 样本中不应包含 'news_features' 键。")
        # else:
        #     print("  样本中不包含 'news_features' 键 (符合预期)。") # 保留此行，但修正为检查 news_features 不存在
        assert 'news_features' not in sample # 断言样本中不包含 'news_features' 键
        print("  样本中不包含 'news_features' 键 (符合预期)。")

        # 检查是否有 NaN 值 (简单检查)
        if torch.isnan(sample['price_seq']).any():
            print("警告: price_seq 中包含 NaN 值")
        if torch.isnan(sample['target_price']).any():
            print("警告: target_price 中包含 NaN 值")
    else:
        print("数据集为空，无法获取样本。")
    
    print("\n--- CryptoDatasetNoNews 测试结束 ---") 