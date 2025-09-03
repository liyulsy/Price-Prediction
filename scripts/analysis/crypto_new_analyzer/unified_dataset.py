from bisect import bisect_left
import torch
import pandas as pd
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import json
import os
import numpy as np

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
        if isinstance(dates, pd.Timestamp):
            dates = pd.DatetimeIndex([dates])
        else:
            raise TypeError("Input `dates` must be a pd.DatetimeIndex, pd.Timestamp, or numpy array of datetime64.")

    features = []
    if freq == 'h':
        features.append(dates.month / 12.0 - 0.5)
        features.append(dates.day / 31.0 - 0.5)
        features.append(dates.dayofweek / 6.0 - 0.5)
        features.append(dates.dayofyear / 365.0 - 0.5)
        features.append(dates.hour / 23.0 - 0.5)
        features.append(dates.isocalendar().week.values.astype(float) / 52.0 - 0.5)
    elif freq == 'd':
        features.append(dates.month / 12.0 - 0.5)
        features.append(dates.day / 31.0 - 0.5)
        features.append(dates.dayofweek / 6.0 - 0.5)
        features.append(dates.dayofyear / 365.0 - 0.5)
        features.append(dates.isocalendar().week.values.astype(float) / 52.0 - 0.5)
    elif freq == 'w':
        features.append(dates.month / 12.0 - 0.5)
        features.append(dates.isocalendar().week.values.astype(float) / 52.0 - 0.5)
    elif freq == 'm':
        features.append(dates.month / 12.0 - 0.5)
    else:
        raise ValueError(f"Unsupported freq: {freq}")
    return np.stack(features, axis=1).astype(np.float32)

class UnifiedCryptoDataset(Dataset):
    def __init__(self, price_data_df, news_data_dict=None, seq_len=24, pred_len=1,
                 processed_news_features_path="processed_news_features.pt", force_recompute_news=False,
                 time_encoding_enabled: bool = True,
                 time_freq: str = 'h',
                 predict_mode: bool = False):
        """
        Args:
            price_data_df: DataFrame, contains price data for all coins, indexed by timestamp, with columns as coin names.
            news_data_dict: Dict[str, List[Dict]] or None. List of news for each coin. If None, news features are not used.
            seq_len: Sequence length, the number of time steps for model input.
            processed_news_features_path: Path to save/load preprocessed news features (used only if news_data_dict is not None).
            force_recompute_news: Whether to force re-computation of news features (used only if news_data_dict is not None).
            time_encoding_enabled: bool, whether to enable time feature encoding.
            time_freq: str, time frequency (e.g., 'h' for hourly), for the time_features function.
            predict_mode: bool. If True, the dataset includes the last possible sequence, which has a dummy target. Useful for inference.
        """
        super().__init__()
        print(f"[UnifiedCryptoDataset] Reading price data: shape={price_data_df.shape}, columns={list(price_data_df.columns)}")
        print(f"[UnifiedCryptoDataset] News data: {'enabled' if news_data_dict is not None else 'disabled'}")
        self.price_data_df = price_data_df
        self.news_data_dict = news_data_dict
        self.seq_len = seq_len
        self.pred_len = pred_len # Store pred_len
        self.time_index = pd.to_datetime(self.price_data_df.index)
        self.time_encoding_enabled = time_encoding_enabled
        self.time_freq = time_freq
        self.predict_mode = predict_mode
        self.has_news = self.news_data_dict is not None

        # 检查时间索引顺序
        if len(self.time_index) > 1:
            time_ascending = self.time_index[0] < self.time_index[-1]
            print(f"[UnifiedCryptoDataset] 时间索引顺序: {'升序(早→晚)' if time_ascending else '降序(晚→早)'}")
            print(f"[UnifiedCryptoDataset] 时间范围: {self.time_index[0]} 到 {self.time_index[-1]}")

            if not time_ascending:
                print(f"⚠️ [UnifiedCryptoDataset] 检测到降序时间索引，新闻特征将按此顺序对齐")

        self.coin_names = list(price_data_df.columns)
        self.coin_to_idx = {name: i for i, name in enumerate(self.coin_names)}
        self.num_coins = len(self.coin_names)
        
        if self.has_news:
            # --- 新闻特征处理 ---
            # 定义新闻特征的总维度 (3个嵌入向量 + 3个相似度分数 + 6个统计特征 + 2个情绪/状态值)
            self.news_feature_dim = 768 * 3 + 3 + 6 + 2
            self.processed_news_features_path = processed_news_features_path
            self.force_recompute_news = force_recompute_news
            self.processed_news_features = None  # 初始化预处理新闻特征为空

            # 检查是否存在预处理好的新闻特征文件，并且不强制重新计算
            if not self.force_recompute_news and self.processed_news_features_path and os.path.exists(self.processed_news_features_path):
                try:
                    # 尝试从缓存文件加载预处理的新闻特征
                    self.processed_news_features = torch.load(self.processed_news_features_path, weights_only=True)
                    print(f"📰 加载缓存新闻特征: {self.processed_news_features.shape}")

                    # 分析每个币种的特征覆盖率
                    print(f"🔍 各币种新闻特征覆盖率分析:")
                    for i, coin_name in enumerate(self.coin_names):
                        if i < self.processed_news_features.shape[1]:
                            coin_features = self.processed_news_features[:, i, :]  # [time, features]

                            # 计算覆盖率统计
                            total_timepoints = coin_features.shape[0]
                            total_features = coin_features.shape[1]

                            # 时间维度覆盖率：有多少时间点有非零特征
                            time_norms = torch.norm(coin_features, dim=1)  # 每个时间点的特征范数
                            active_timepoints = (time_norms > 0).sum().item()
                            time_coverage = active_timepoints / total_timepoints

                            # 特征维度覆盖率：有多少特征维度被使用
                            feature_activity = (coin_features != 0).any(dim=0)  # 每个特征维度是否被使用
                            active_features = feature_activity.sum().item()
                            feature_coverage = active_features / total_features

                            # 整体非零比例
                            nonzero_ratio = (coin_features != 0).float().mean().item()

                            # 特征强度统计
                            feature_mean = coin_features.mean().item()
                            feature_std = coin_features.std().item()
                            feature_norm = torch.norm(coin_features).item()

                            print(f"  {coin_name}:")
                            print(f"    时间覆盖率: {time_coverage:.1%} ({active_timepoints}/{total_timepoints})")
                            print(f"    特征覆盖率: {feature_coverage:.1%} ({active_features}/{total_features})")
                            print(f"    非零比例: {nonzero_ratio:.1%}")
                            print(f"    特征强度: 均值={feature_mean:.6f}, 标准差={feature_std:.6f}, 范数={feature_norm:.4f}")
                        else:
                            print(f"  {coin_name}: ❌ 索引超出范围")

                    # 不在这里验证形状，而是在后面尝试自动对齐
                except Exception as e:
                    # 如果加载失败，也置为空，后续将重新计算
                    print(f"❌ 加载新闻特征失败: {e}")
                    self.processed_news_features = None
            
            # 如果预处理的新闻特征仍然为空（因为文件不存在、加载失败或强制重新计算）
            if self.processed_news_features is None:
                print("🔄 重新计算新闻特征...")
                # 调用内部方法来准备/计算所有新闻特征
                self.processed_news_features = self._prepare_all_news_features()
                # 如果指定了保存路径，则将新计算的特征保存到文件以备后用
                if self.processed_news_features_path:
                    save_dir = os.path.dirname(self.processed_news_features_path)
                    if save_dir and not os.path.exists(save_dir):
                        os.makedirs(save_dir)  # 确保目录存在
                    torch.save(self.processed_news_features, self.processed_news_features_path)
            else:
                # 检查新闻特征与价格数据的时间索引是否匹配
                if self.processed_news_features.shape[0] != len(self.time_index):
                    print(f"⚠️  新闻特征时间维度 ({self.processed_news_features.shape[0]}) 与价格数据 ({len(self.time_index)}) 不匹配")
                    alignment_success = False

                    if self.processed_news_features.shape[0] == len(self.time_index) + 1:
                        # 新闻特征比价格数据多一个时间点（通常是diff/pct_change导致的）
                        print("🔧 自动对齐：删除新闻特征的第一个时间点")
                        self.processed_news_features = self.processed_news_features[1:]
                        print(f"✅ 对齐成功：新闻特征形状 {self.processed_news_features.shape}")
                        alignment_success = True
                    elif self.processed_news_features.shape[0] == len(self.time_index) - 1:
                        # 价格数据比新闻特征多一个时间点
                        print("🔧 自动对齐：在新闻特征开头补零")
                        padding = torch.zeros(1, self.num_coins, self.news_feature_dim, dtype=torch.float32)
                        self.processed_news_features = torch.cat([padding, self.processed_news_features], dim=0)
                        print(f"✅ 对齐成功：新闻特征形状 {self.processed_news_features.shape}")
                        alignment_success = True

                    if not alignment_success:
                        print(f"❌ 无法自动对齐，时间维度差异过大: {self.processed_news_features.shape[0]} vs {len(self.time_index)}")
                        print(f"� 尝试重新计算新闻特征...")
                        # 自动对齐失败，重新计算新闻特征
                        self.processed_news_features = self._prepare_all_news_features()
                        # 保存重新计算的特征
                        if self.processed_news_features_path:
                            save_dir = os.path.dirname(self.processed_news_features_path)
                            if save_dir and not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            torch.save(self.processed_news_features, self.processed_news_features_path)
                            print(f"💾 已保存重新计算的新闻特征到: {self.processed_news_features_path}")
                else:
                    print(f"✅ 新闻特征与价格数据时间维度匹配: {self.processed_news_features.shape[0]}")
        else:
            # 如果不使用新闻数据
            self.news_feature_dim = 0  # 将新闻特征维度设为0
            # 创建一个形状正确但第三维为0的空张量，以保持数据结构和类型的一致性
            self.processed_news_features = torch.zeros(len(self.time_index), self.num_coins, 0, dtype=torch.float32)

        # --- 时间特征编码 ---
        if self.time_encoding_enabled:
            # 如果启用时间编码，则为所有时间点生成时间特征
            encoded_stamps_np = time_features(self.time_index, freq=self.time_freq)
            self.all_time_stamps_encoded = torch.tensor(encoded_stamps_np, dtype=torch.float32)
        else:
            # 如果不启用时间编码，创建一个全零的张量作为占位符
            # 先用一个虚拟日期计算出时间特征应有的维度
            _dummy_dates = pd.DatetimeIndex([pd.Timestamp('2000-01-01')])
            num_time_features_dim = time_features(_dummy_dates, freq=self.time_freq).shape[1]
            # 创建一个形状为 (时间点数量, 特征维度) 的全零张量
            self.all_time_stamps_encoded = torch.zeros(len(self.time_index), num_time_features_dim, dtype=torch.float32)
        # 记录每个时间点实际拥有的时间特征数量（无论是真实计算的还是0）
        self.num_actual_time_features = self.all_time_stamps_encoded.shape[1]

    def _extract_single_news_feature(self, news_item):
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
                news_item.get('title_length', 0), news_item.get('title_words', 0),
                news_item.get('subtitle_length', 0), news_item.get('subtitle_words', 0),
                news_item.get('body_length', 0), news_item.get('body_words', 0)
            ], dtype=torch.float32)
            
            sentiment_status = torch.tensor([
                news_item.get('sentiment_value', 0.0),
                news_item.get('status_value', 0.0)
            ], dtype=torch.float32)

            return torch.cat([title_emb, subtitle_emb, body_emb, similarities, stats, sentiment_status])
        except Exception:
            return torch.zeros(self.news_feature_dim, dtype=torch.float32)

    def _process_coin_news(self):
        coin_news_indices = {}
        for coin_name, news_list in self.news_data_dict.items():
            if coin_name not in self.coin_to_idx:
                continue

            valid_news_items = [item for item in news_list if pd.to_datetime(item.get('published_ts'), errors='coerce') is not pd.NaT]
            
            sorted_news = sorted(valid_news_items, key=lambda x: pd.to_datetime(x['published_ts']))
            times = [pd.to_datetime(n['published_ts']) for n in sorted_news]
            
            coin_news_indices[coin_name] = {'news': sorted_news, 'times': times, 'count': len(sorted_news)}
        return coin_news_indices

    def _prepare_all_news_features(self):
        T = len(self.time_index)
        all_features = torch.zeros(T, self.num_coins, self.news_feature_dim, dtype=torch.float32)
        coin_news_data = self._process_coin_news()
        
        default_params = {'validity_days': 10, 'decay_factor': 1.0}

        for t, timestamp in enumerate(self.time_index):
            for coin_name in self.coin_names:
                coin_idx = self.coin_to_idx[coin_name]
                
                if coin_name not in coin_news_data or not coin_news_data[coin_name]['times']:
                    continue

                current_coin_data = coin_news_data[coin_name]
                params = default_params
                validity_days = params['validity_days']

                search_start_time_cutoff = timestamp - pd.Timedelta(days=validity_days + 1)
                start_index = bisect_left(current_coin_data['times'], search_start_time_cutoff)

                valid_news_info = []
                for i in range(start_index, len(current_coin_data['times'])):
                    news_time = current_coin_data['times'][i]
                    if news_time > timestamp: break
                    time_diff_days = (timestamp - news_time).days
                    if time_diff_days <= validity_days:
                        weight = 1.0 - (time_diff_days / float(validity_days))
                        valid_news_info.append((current_coin_data['news'][i], weight))
                
                if valid_news_info:
                    features_tensor_list = [self._extract_single_news_feature(item) for item, _ in valid_news_info]
                    weights_list = [weight for _, weight in valid_news_info]
                    
                    features_stack = torch.stack(features_tensor_list)
                    weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
                    
                    if weights_tensor.sum() > 0:
                        weights_tensor = weights_tensor / weights_tensor.sum()
                        all_features[t, coin_idx] = (features_stack * weights_tensor.unsqueeze(1)).sum(dim=0)
                    elif features_stack.numel() > 0:
                        all_features[t, coin_idx] = features_stack.mean(dim=0)
        return all_features

    def __len__(self):
        # The number of possible start indices for a complete sample (input sequence + target sequence)
        if self.predict_mode:
            # In predict mode, we only need a valid input sequence
            return len(self.time_index) - self.seq_len + 1
        else:
            # In training/validation mode, we need a valid input and target sequence
            return len(self.time_index) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq_end_idx = idx + self.seq_len
        target_end_idx = seq_end_idx + self.pred_len

        price_seq_df = self.price_data_df[self.coin_names].iloc[idx:seq_end_idx]
        price_seq_tensor = torch.tensor(price_seq_df.values, dtype=torch.float32)

        if target_end_idx <= len(self.price_data_df):
            # We have a valid target sequence
            target_price_df = self.price_data_df[self.coin_names].iloc[target_end_idx]
            target_price_tensor = torch.tensor(target_price_df.values, dtype=torch.float32)
        else:
            # This handles cases at the end of the dataset, including predict_mode where a dummy target is needed.
            target_price_tensor = torch.zeros(self.num_coins, dtype=torch.float32)

        price_seq_mark = self.all_time_stamps_encoded[idx:seq_end_idx]

        return_dict = {
            'price_seq': price_seq_tensor,
            'price_seq_mark': price_seq_mark,
            'target_price': target_price_tensor
        }

        if self.has_news:
            news_features_tensor = self.processed_news_features[seq_end_idx-1]
            return_dict['news_features'] = news_features_tensor
        
        return return_dict

def load_news_data(features_dir, coin_names):
    news_data_dict = {}
    for coin in coin_names:
        file_path = os.path.join(features_dir, f"{coin.replace(' ', '')}_features.json")
        try:
            with open(file_path, 'r') as f:
                news_data_dict[coin] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            news_data_dict[coin] = []
    return news_data_dict

if __name__ == '__main__':
    COIN_NAMES = ['BTC', 'ETH', 'BNB']
    SEQ_LEN = 180
    
    date_rng = pd.date_range(start='2023-01-01', end='2024-01-01', freq='H')
    simulated_price_data = pd.DataFrame(torch.randn(len(date_rng), len(COIN_NAMES)).numpy(), index=date_rng, columns=COIN_NAMES)
    
    features_folder = 'features' # Assume a local features folder for testing
    if not os.path.exists(features_folder): os.makedirs(features_folder)
    news_data = load_news_data(features_folder, COIN_NAMES)

    print("\n--- Test Case 1: With News Data ---")
    dataset_with_news = UnifiedCryptoDataset(simulated_price_data, news_data, seq_len=SEQ_LEN)
    print(f"Dataset with news size: {len(dataset_with_news)}")
    if len(dataset_with_news) > 0:
        sample = dataset_with_news[0]
        assert 'news_features' in sample

    print("\n--- Test Case 2: Without News Data ---")
    dataset_no_news = UnifiedCryptoDataset(simulated_price_data, news_data_dict=None, seq_len=SEQ_LEN)
    print(f"Dataset without news size: {len(dataset_no_news)}")
    if len(dataset_no_news) > 0:
        sample = dataset_no_news[0]
        assert 'news_features' not in sample

    print("\n--- Test Case 3: Predict Mode ---")
    dataset_predict = UnifiedCryptoDataset(simulated_price_data, news_data_dict=None, seq_len=SEQ_LEN, predict_mode=True)
    expected_len = len(simulated_price_data) - SEQ_LEN + 1
    print(f"Dataset in predict mode size: {len(dataset_predict)} (Expected: {expected_len})")
    assert len(dataset_predict) == expected_len
    if len(dataset_predict) > 0:
        last_sample = dataset_predict[len(dataset_predict)-1]
        assert torch.all(last_sample['target_price'] == 0)
    
    print("\n--- All tests passed! ---") 