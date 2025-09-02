from bisect import bisect_left
import torch
import pandas as pd
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import json
import os
import numpy as np # Added for time_features

# Copied time_features function here for now
# Ideally, this should be in a shared utils.py
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

class CryptoDataset(Dataset):
    def __init__(self, price_data_df, news_data_dict, seq_len=24, processed_news_features_path="processed_news_features.pt", force_recompute_news=False,
                 time_encoding_enabled: bool = True, 
                 time_freq: str = 'h'):
        """
        Args:
            price_data_df: DataFrame, 包含所有币种的价格数据，索引为时间戳，列为币种名称。
            news_data_dict: Dict[str, List[Dict]], 每个币种的新闻列表。
                            key: 币种名称 (e.g., 'BTC', 'ETH')
                            value: 该币种的新闻列表，每个元素是包含新闻特征的字典。
            seq_len: 序列长度，模型输入的时间步数。
            processed_news_features_path: 预处理新闻特征的保存路径。
            force_recompute_news: 是否强制重新计算新闻特征。
            time_encoding_enabled: bool, 是否启用时间特征编码。
            time_freq: str, 时间频率 (e.g., 'h' for hourly), 用于 time_features 函数。
        """
        super().__init__()
        self.price_data_df = price_data_df
        self.news_data_dict = news_data_dict
        self.seq_len = seq_len
        self.time_index = pd.to_datetime(self.price_data_df.index)
        self.time_encoding_enabled = time_encoding_enabled
        self.time_freq = time_freq

        # 定义币种顺序和映射，确保一致性
        self.coin_names = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
        self.coin_to_idx = {name: i for i, name in enumerate(self.coin_names)}
        self.num_coins = len(self.coin_names)
        
        # 新闻特征维度 (3个768维嵌入 + 3个相似度 + 6个统计特征 + 2个情感/状态值)
        self.news_feature_dim = 768 * 3 + 3 + 6 + 2

        self.processed_news_features_path = processed_news_features_path
        self.force_recompute_news = force_recompute_news
        self.processed_news_features = None # 初始化

        if not self.force_recompute_news and self.processed_news_features_path and os.path.exists(self.processed_news_features_path):
            try:
                print(f"尝试从 {self.processed_news_features_path} 加载预处理的新闻特征...")
                self.processed_news_features = torch.load(self.processed_news_features_path, weights_only=True)
                expected_shape = (len(self.time_index), self.num_coins, self.news_feature_dim)
                if self.processed_news_features.shape == expected_shape:
                    print(f"已成功加载预处理的新闻特征。Shape: {self.processed_news_features.shape}")
                else:
                    print(f"警告: 加载的特征形状 {self.processed_news_features.shape} 与预期形状 {expected_shape} 不符。将重新计算。")
                    self.processed_news_features = None # 置空以触发重新计算
            except Exception as e:
                print(f"加载预处理新闻特征失败: {e}。将重新计算。")
                self.processed_news_features = None # 置空以触发重新计算
        
        if self.processed_news_features is None: # 如果需要计算 (未加载成功或强制重新计算)
            print("正在计算新闻特征...")
            self.processed_news_features = self._prepare_all_news_features()
            if self.processed_news_features_path:
                # 确保保存路径的目录存在
                save_dir = os.path.dirname(self.processed_news_features_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f"已创建目录: {save_dir}")
                try:
                    print(f"正在将计算出的新闻特征保存至 {self.processed_news_features_path}...")
                    torch.save(self.processed_news_features, self.processed_news_features_path)
                    print("新闻特征保存成功。")
                except Exception as e:
                    print(f"保存新闻特征失败: {e}")

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

    def _extract_single_news_feature(self, news_item):
        """从单个新闻字典中提取并拼接所有数值特征。"""
        try:
            title_emb = torch.tensor(news_item['title_embedding'][0], dtype=torch.float32)
            subtitle_emb = torch.tensor(news_item['subtitle_embedding'][0], dtype=torch.float32)
            body_emb = torch.tensor(news_item['body_embedding'][0], dtype=torch.float32)

            similarities = torch.tensor([
                news_item.get('title_subtitle_similarity', 0.0),
                news_item.get('title_body_similarity', 0.0), 
                news_item.get('subtitle_body_similarity', 0.0)
            ], dtype=torch.float32)

            stats = torch.tensor([
                news_item.get('title_length', 0),
                news_item.get('title_words', 0),
                news_item.get('subtitle_length', 0),
                news_item.get('subtitle_words', 0),
                news_item.get('body_length', 0),
                news_item.get('body_words', 0)
            ], dtype=torch.float32)
            
            sentiment_status = torch.tensor([
                news_item.get('sentiment_value', 0.0), # 默认为0.0（中性或未知）
                news_item.get('status_value', 0.0)    # 默认为0.0（中性或未知）
            ], dtype=torch.float32)

            return torch.cat([title_emb, subtitle_emb, body_emb, similarities, stats, sentiment_status])
        except Exception as e:
            print(f"提取新闻特征时出错: {news_item.get('title', '未知标题')} - {e}")
            return torch.zeros(self.news_feature_dim, dtype=torch.float32)


    def _process_coin_news(self):
        """
        对每个币种的新闻数据进行预处理和排序。
        """
        coin_news_indices = {}
        for coin_name, news_list in self.news_data_dict.items():
            if coin_name not in self.coin_to_idx:
                print(f"警告: 新闻数据中的币种 {coin_name} 不在预定义币种列表中，将被忽略。")
                continue

            news_count = len(news_list)
            # print(f"处理 {coin_name} 的新闻数据，共 {news_count} 条")

            valid_news_items = []
            for item in news_list:
                try:
                    # 确保时间戳有效
                    pd.to_datetime(item['published_ts'])
                    valid_news_items.append(item)
                except Exception as e:
                    # print(f"警告: 币种 {coin_name} 的新闻条目时间戳格式无效，已跳过: {item.get('source_created_time')} - {e}")
                    pass
            
            sorted_news = sorted(
                valid_news_items,
                key=lambda x: pd.to_datetime(x['published_ts'])
            )
            times = [pd.to_datetime(n['published_ts']) for n in sorted_news]
            
            coin_news_indices[coin_name] = {
                'news': sorted_news,
                'times': times,
                'count': len(sorted_news)
            }
        return coin_news_indices

    def _prepare_all_news_features(self):
        """预处理所有时间点、所有币种的新闻特征。"""
        T = len(self.time_index)
        # [时间点数, 币种数, 特征维度]
        all_features = torch.zeros(T, self.num_coins, self.news_feature_dim, dtype=torch.float32)
        
        coin_news_data = self._process_coin_news()

        # 为不同币种设置不同的有效期（天）和衰减策略
        coin_specific_params = {
            'BTC': {'validity_days': 10, 'decay_factor': 1.0},    # 新闻充足
            'ETH': {'validity_days': 10, 'decay_factor': 1.0},    # 新闻充足
            'BNB': {'validity_days': 10, 'decay_factor': 1.0},   # 新闻极少，最晚2023-06, 配合填充
            'XRP': {'validity_days': 10, 'decay_factor': 1.0},    # 新闻量尚可
            'LTC': {'validity_days': 10, 'decay_factor': 1.0},   # 新闻偏少
            'DOGE': {'validity_days': 10, 'decay_factor': 1.0},  # 新闻量一般
            'SOL': {'validity_days': 10, 'decay_factor': 1.0},    # 新闻量尚可
            'AVAX': {'validity_days': 10, 'decay_factor': 1.0}  # 新闻很少
        }
        # 默认参数 (如果未来有新币种未在此明确定义)
        default_params = {'validity_days': 10, 'decay_factor': 1.0}

        for t, timestamp in enumerate(self.time_index):
            for coin_name in self.coin_names:
                coin_idx = self.coin_to_idx[coin_name]
                
                if coin_name not in coin_news_data or not coin_news_data[coin_name]['times']:
                    continue # 该币种无新闻数据

                current_coin_data = coin_news_data[coin_name]
                params = coin_specific_params.get(coin_name, default_params)
                validity_days = params['validity_days']
                decay_factor = params['decay_factor']

                # 计算一个大致的起始搜索时间点，任何早于此 cutoff 的新闻肯定无效
                search_start_time_cutoff = timestamp - pd.Timedelta(days=validity_days + 1)
                start_index = bisect_left(current_coin_data['times'], search_start_time_cutoff)

                # 找到该时间点之前的所有有效新闻
                valid_news_info = [] # (news_item, weight)

                for i in range(start_index, len(current_coin_data['times'])):
                    news_time = current_coin_data['times'][i]

                    if news_time > timestamp:
                        break # 优化1: 如果新闻时间已经超过当前时间戳，后续无需再找

                    # 应用原始的基于天的有效性判断
                    time_diff_days = (timestamp - news_time).days
                    if time_diff_days <= validity_days:
                        # weight = 1.0 / (1.0 + decay_factor * time_diff_days)
                        weight = 1.0 - (time_diff_days / float(validity_days))
                        # print(f"新闻时间: {news_time}, 权重: {weight}")
                        
                        valid_news_info.append((current_coin_data['news'][i], weight))
                
                if valid_news_info:
                    if len(valid_news_info) == 1:
                        news_item, _ = valid_news_info[0]
                        all_features[t, coin_idx] = self._extract_single_news_feature(news_item)
                    else:
                        features_tensor_list = []
                        weights_list = []
                        for news_item, weight in valid_news_info:
                            features_tensor_list.append(self._extract_single_news_feature(news_item))
                            weights_list.append(weight)
                        
                        features_stack = torch.stack(features_tensor_list)
                        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
                        if weights_tensor.sum() > 0 : #避免除以0
                             weights_tensor = weights_tensor / weights_tensor.sum() # 归一化权重
                             all_features[t, coin_idx] = (features_stack * weights_tensor.unsqueeze(1)).sum(dim=0)
                        elif features_stack.numel() > 0 : #如果权重和为0但有特征，则取平均
                             all_features[t, coin_idx] = features_stack.mean(dim=0)


        # # 步骤2: 填充空缺的新闻特征 (可选，但对新闻稀疏的币种有帮助)
        # all_features = self._fill_missing_with_related(all_features, coin_news_data)
        
        # 打印新闻覆盖率统计
        print("\n新闻特征覆盖率统计 (填充后):")
        for i, coin in enumerate(self.coin_names):
            non_zero_count = (all_features[:, i].abs().sum(dim=1) > 1e-6).sum().item() # 检查非零行
            total_time_steps = all_features.shape[0]
            coverage = (non_zero_count / total_time_steps) * 100 if total_time_steps > 0 else 0
            print(f"{coin}: {non_zero_count}/{total_time_steps} ({coverage:.2f}%) 个时间点有新闻特征")
            
        return all_features

    def _fill_missing_with_related(self, processed_features, coin_news_data):
        """
        用相关币种的新闻填充没有新闻的币种的特征。
        这部分逻辑可以根据实际的相关性定义进行扩展。
        """
        # 定义币种相关性 和 填充权重
        # 示例: BNB 可能与 BTC/ETH 相关, AVAX 与 ETH 相关
        coin_correlation = {
            'BNB': [('BTC', 0.6), ('ETH', 0.5)], # BNB 与 BTC/ETH 相关性较高
            'AVAX': [('ETH', 0.6), ('SOL', 0.4)], # AVAX 与 ETH/SOL 相关
            'LTC': [('BTC', 0.5)],             # LTC 可能与 BTC 相关
            'DOGE': [('BTC', 0.4)],            # DOGE 可能受 BTC 影响
            # 对于其他新闻也可能稀疏的，可以补充
        }
        # 对于新闻极度稀疏的币，如果在上面没有定义特定相关性，或特定相关币也无新闻，则使用市场平均（例如BTC或ETH）
        fallback_coins = [('BTC', 0.4), ('ETH', 0.3)] # 默认回退到BTC或ETH


        T = processed_features.shape[0]
        for t in range(T):
            for target_coin_name in self.coin_names:
                target_idx = self.coin_to_idx[target_coin_name]
                # 检查该币种在该时间点是否已经有新闻 (特征向量不为零)
                if processed_features[t, target_idx].abs().sum() < 1e-6: # 如果是全零向量
                    
                    source_found = False
                    correlations_to_try = coin_correlation.get(target_coin_name, fallback_coins)

                    for related_coin_name, fill_weight in correlations_to_try:
                        if related_coin_name in self.coin_to_idx:
                            related_idx = self.coin_to_idx[related_coin_name]
                            # 如果相关币种有新闻特征
                            if processed_features[t, related_idx].abs().sum() > 1e-6:
                                processed_features[t, target_idx] = processed_features[t, related_idx] * fill_weight
                                source_found = True
                                break # 找到一个相关币种就填充
                    
                    # 如果在定义的强相关币种中没找到，对于新闻极少的币，可以尝试用市场领导者（如BTC）填充
                    if not source_found and coin_news_data.get(target_coin_name, {}).get('count', 0) < 20: #少于20条新闻的币
                         for fallback_coin_name, fallback_weight in fallback_coins:
                            fallback_idx = self.coin_to_idx[fallback_coin_name]
                            if processed_features[t, fallback_idx].abs().sum() > 1e-6:
                                processed_features[t, target_idx] = processed_features[t, fallback_idx] * fallback_weight
                                break
        return processed_features

    def __len__(self):
        # 数据集的长度是总时间步数减去序列长度，因为每个样本都需要seq_len的历史数据
        return len(self.time_index) - self.seq_len

    def __getitem__(self, idx):
        # 结束索引是 idx + seq_len
        end_idx = idx + self.seq_len
        # 目标是end_idx的下一个时间点 (如果需要预测未来)
        # target_idx = end_idx 

        # 获取价格序列: 从 idx 到 end_idx-1
        # price_data_df的列应该是self.coin_names顺序
        price_seq_df = self.price_data_df[self.coin_names].iloc[idx:end_idx]
        price_seq_tensor = torch.tensor(price_seq_df.values, dtype=torch.float32)
        
        # 获取目标时间点的新闻特征 (对应价格序列的最后一个点)
        # processed_news_features 的索引对应 self.time_index
        # 所以，价格序列的最后一个点是 self.time_index[end_idx-1]
        news_features_tensor = self.processed_news_features[end_idx-1] # Shape: [num_coins, news_feature_dim]
        
        # 获取目标价格 (价格序列之后的一个时间点)
        # 确保 target_idx 不会超出 price_data_df 的范围
        if end_idx < len(self.price_data_df):
            target_price_series = self.price_data_df[self.coin_names].iloc[end_idx]
            target_price_tensor = torch.tensor(target_price_series.values, dtype=torch.float32)
        else:
            # 如果是最后一个可能的序列，没有未来价格点可作为目标，可以返回序列最后一个价格或特定值
            target_price_tensor = torch.zeros(self.num_coins, dtype=torch.float32) # 或者其他处理方式

        # Get encoded time features for the price sequence
        price_seq_mark = self.all_time_stamps_encoded[idx : idx + self.seq_len]
        # Target mark might be needed if model has explicit decoder input for time marks
        # target_price_mark = self.all_time_stamps_encoded[idx + self.seq_len]

        return {
            'price_seq': price_seq_tensor,      # Shape: [seq_len, num_coins, (price_feature_dim)] - 这里假设价格数据直接是数值
            'price_seq_mark': price_seq_mark,    # Shape: [seq_len, num_actual_time_features]
            'news_features': news_features_tensor, # Shape: [num_coins, news_feature_dim]
            'target_price': target_price_tensor  # Shape: [num_coins]
        }

# --- 辅助函数和使用示例 ---
def load_news_data(features_dir, coin_names):
    """加载所有币种的新闻数据"""
    news_data_dict = {}
    for coin in coin_names:
        file_path = os.path.join(features_dir, f"{coin.replace(' ', '')}_features.json") # 移除币安币的空格
        try:
            with open(file_path, 'r') as f:
                news_data_dict[coin] = json.load(f)
        except FileNotFoundError:
            print(f"警告: 未找到 {coin} 的新闻文件: {file_path}")
            news_data_dict[coin] = []
        except json.JSONDecodeError:
            print(f"警告: 解析 {coin} 的新闻文件失败: {file_path}")
            news_data_dict[coin] = []
    return news_data_dict

if __name__ == '__main__':
    # --- 0. 定义参数 ---
    COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
    SEQ_LEN = 180 # 使用过去N小时/天的数据，根据数据频率调整
    
    # --- 1. 准备价格数据 (示例) ---
    # 确保时间索引是 pd.DatetimeIndex
    # 列名应与COIN_NAMES中的币种名称一致且顺序一致
    date_rng = pd.date_range(start='2022-12-22 01:00:00', end='2024-12-10 23:00:00', freq='H')
    num_timesteps = len(date_rng)
    
    # 模拟价格数据: [num_timesteps, num_coins]
    # 实际应用中，你需要从你的 '1H.csv' 或其他来源加载价格数据
    # 并确保其形状和列名正确
    simulated_price_data = pd.DataFrame(
        torch.randn(num_timesteps, len(COIN_NAMES)).numpy(),
        index=date_rng,
        columns=COIN_NAMES 
    )
    print("模拟价格数据:")
    print(simulated_price_data.head())

    # --- 2. 加载新闻数据 ---
    # 注意：请将 "path/to/your/features_folder" 替换为实际的features文件夹路径
    features_folder = '/mnt/sda/ly/research/Project1/crypto_new_analyzer/features' # 请替换为您的实际路径
    news_data = load_news_data(features_folder, COIN_NAMES)
    
    # 简单检查加载的新闻数据
    for coin, data_list in news_data.items():
        print(f"{coin} 加载了 {len(data_list)} 条新闻.")
        if data_list:
            print(f"  第一条新闻示例 ('published_ts'): {data_list[0].get('published_ts')}")

    # --- 3. 创建CryptoDataset实例 ---
    print("\n--- 创建 CryptoDataset 实例 ---")
    # 示例：使用缓存功能
    # 若要强制重新计算并更新缓存，设置 force_recompute_news=True
    # 第一次运行时，由于缓存文件不存在，会自动计算并保存。
    # 后续运行，如果文件存在且 force_recompute_news=False，则会加载缓存。
    dataset = CryptoDataset(
        simulated_price_data, 
        news_data, 
        seq_len=SEQ_LEN,
        processed_news_features_path="cache/all_processed_news_feature_10days.pt", # 可以自定义路径和文件名
        force_recompute_news=False,
        time_encoding_enabled=True,
        time_freq='h'
    )

    # --- 4. 验证数据集 ---
    print(f"\n数据集大小: {len(dataset)}")
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n获取的第一个样本:")
        print(f"  价格序列 (price_seq) shape: {sample['price_seq'].shape}")
        print(f"  新闻特征 (news_features) shape: {sample['news_features'].shape}")
        print(f"  目标价格 (target_price) shape: {sample['target_price'].shape}")

        print(f"\n价格序列的前5行 (第一个币):")
        print(sample['price_seq'][:5, 0])
        print(f"\n新闻特征 (第一个币) 的前10个值:")
        print(sample['news_features'][0, :10])
        print(f"\n目标价格 (第一个币):")
        print(sample['target_price'][0])
        
        # 检查是否有 NaN 值
        if torch.isnan(sample['price_seq']).any():
            print("警告: price_seq 中包含 NaN 值")
        if torch.isnan(sample['news_features']).any():
            print("警告: news_features 中包含 NaN 值")
        if torch.isnan(sample['target_price']).any():
            print("警告: target_price 中包含 NaN 值")

    else:
        print("数据集为空，无法获取样本。请检查价格数据的时间范围和序列长度。")

    # --- 5. (可选) 创建 DataLoader ---
    # from torch.utils.data import DataLoader
    # train_loader = DataLoader(crypto_dataset, batch_size=32, shuffle=True)
    # print("\n成功创建 DataLoader.") 