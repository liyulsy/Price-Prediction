import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# from .base_loader import BaseLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def generate_edge_index(input_series, threshold=0.6, return_weights=False):
    """
    基于相关系数矩阵构建图的边
    Args:
        input_series: DataFrame，行是时间，列是币种价格
        threshold: float，相关系数阈值，绝对值大于此值才保留边
        return_weights: bool，是否返回边权重（相关系数值）
    
    Returns:
        edge_index: [2, num_edges] torch.LongTensor
        edge_weights: [num_edges] torch.FloatTensor（可选）
    """
    correlation_matrix = input_series.corr()
    edges = []
    weights = []

    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > threshold:
                edges.append([i, j])
                edges.append([j, i])
                weights.append(abs(corr))
                weights.append(abs(corr))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if return_weights:
        edge_weights = torch.tensor(weights, dtype=torch.float)
        return edge_index, edge_weights
    else:
        return edge_index

def generate_advanced_edge_index(input_series, method='multi_layer', **kwargs):
    """
    高级图构建方法

    Args:
        input_series: 价格数据 DataFrame
        method: 构图方法
            - 'multi_layer': 多层图结构
            - 'dynamic': 动态时变图
            - 'domain_knowledge': 基于领域知识的图
            - 'attention_based': 基于注意力的图
        **kwargs: 各种方法的参数

    Returns:
        edge_index: 边索引
        edge_weights: 边权重（可选）
    """
    if method == 'multi_layer':
        return _generate_multi_layer_graph(input_series, **kwargs)
    elif method == 'dynamic':
        return _generate_dynamic_graph(input_series, **kwargs)
    elif method == 'domain_knowledge':
        return _generate_domain_knowledge_graph(input_series, **kwargs)
    elif method == 'attention_based':
        return _generate_attention_graph(input_series, **kwargs)
    else:
        raise ValueError(f"Unknown graph construction method: {method}")

def _generate_multi_layer_graph(input_series,
                               correlation_threshold=0.3,
                               volatility_threshold=0.5,
                               trend_threshold=0.4):
    """
    多层图结构：结合相关性、波动性相似性、趋势相似性
    """
    n_nodes = len(input_series.columns)
    edges = []
    edge_weights = []

    # 1. 相关性层
    corr_matrix = input_series.corr().abs()

    # 2. 波动性相似性层
    volatility = input_series.rolling(window=24).std()  # 24小时滚动标准差
    volatility_sim = cosine_similarity(volatility.fillna(0).T)

    # 3. 趋势相似性层
    returns = input_series.pct_change().fillna(0)
    trend_sim = cosine_similarity(returns.T)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # 计算综合权重
            corr_weight = corr_matrix.iloc[i, j] if corr_matrix.iloc[i, j] > correlation_threshold else 0
            vol_weight = volatility_sim[i, j] if volatility_sim[i, j] > volatility_threshold else 0
            trend_weight = abs(trend_sim[i, j]) if abs(trend_sim[i, j]) > trend_threshold else 0

            # 综合权重（可以调整权重比例）
            combined_weight = 0.4 * corr_weight + 0.3 * vol_weight + 0.3 * trend_weight

            if combined_weight > 0.2:  # 最终阈值
                edges.extend([[i, j], [j, i]])
                edge_weights.extend([combined_weight, combined_weight])

    if not edges:
        # 如果没有边，创建一个最小连通图
        for i in range(n_nodes - 1):
            edges.extend([[i, i+1], [i+1, i]])
            edge_weights.extend([0.1, 0.1])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    return edge_index, edge_weights

def _generate_dynamic_graph(input_series, window_size=168, overlap=24):
    """
    动态时变图：基于滑动窗口的时变相关性

    Args:
        window_size: 滑动窗口大小（小时）
        overlap: 窗口重叠大小
    """
    n_nodes = len(input_series.columns)
    all_edges = []
    all_weights = []

    # 滑动窗口计算时变相关性
    for start_idx in range(0, len(input_series) - window_size, overlap):
        end_idx = start_idx + window_size
        window_data = input_series.iloc[start_idx:end_idx]

        if len(window_data) < window_size // 2:  # 确保有足够数据
            continue

        corr_matrix = window_data.corr().abs()

        # 动态阈值：基于当前窗口的相关性分布
        threshold = np.percentile(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)], 70)

        window_edges = []
        window_weights = []

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if corr_matrix.iloc[i, j] > threshold:
                    window_edges.extend([[i, j], [j, i]])
                    window_weights.extend([corr_matrix.iloc[i, j], corr_matrix.iloc[i, j]])

        all_edges.extend(window_edges)
        all_weights.extend(window_weights)

    # 统计边的出现频率，作为最终权重
    edge_counts = {}
    edge_weight_sums = {}

    for i in range(0, len(all_edges), 2):  # 每两个边为一对（无向图）
        edge = tuple(sorted([all_edges[i][0], all_edges[i][1]]))
        weight = all_weights[i]

        if edge not in edge_counts:
            edge_counts[edge] = 0
            edge_weight_sums[edge] = 0

        edge_counts[edge] += 1
        edge_weight_sums[edge] += weight

    # 选择出现频率高的边
    min_frequency = max(1, len(range(0, len(input_series) - window_size, overlap)) // 4)

    final_edges = []
    final_weights = []

    for edge, count in edge_counts.items():
        if count >= min_frequency:
            avg_weight = edge_weight_sums[edge] / count
            final_edges.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
            final_weights.extend([avg_weight, avg_weight])

    if not final_edges:
        # 创建最小连通图
        for i in range(n_nodes - 1):
            final_edges.extend([[i, i+1], [i+1, i]])
            final_weights.extend([0.1, 0.1])

    edge_index = torch.tensor(final_edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(final_weights, dtype=torch.float32)

    return edge_index, edge_weights

def _generate_domain_knowledge_graph(input_series, coin_names=None):
    """
    基于领域知识的图构建：考虑加密货币的实际关联性
    """
    if coin_names is None:
        coin_names = input_series.columns.tolist()

    # 定义加密货币的类别和关系
    crypto_categories = {
        'major': ['BTC', 'ETH'],  # 主要币种
        'altcoins': ['BNB', 'XRP', 'LTC', 'DOGE'],  # 山寨币
        'defi': ['SOL', 'AVAX'],  # DeFi相关
    }

    # 市值排名权重（假设的排名）
    market_cap_ranks = {
        'BTC': 1, 'ETH': 2, 'BNB': 3, 'XRP': 4,
        'SOL': 5, 'DOGE': 6, 'LTC': 7, 'AVAX': 8
    }

    n_nodes = len(coin_names)
    edges = []
    edge_weights = []

    # 计算基础相关性
    corr_matrix = input_series.corr().abs()

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            coin_i = coin_names[i]
            coin_j = coin_names[j]

            # 基础相关性权重
            base_weight = corr_matrix.iloc[i, j]

            # 领域知识加权
            domain_weight = 0

            # 1. 同类别加权
            for category, coins in crypto_categories.items():
                if coin_i in coins and coin_j in coins:
                    domain_weight += 0.3

            # 2. 市值相近加权
            if coin_i in market_cap_ranks and coin_j in market_cap_ranks:
                rank_diff = abs(market_cap_ranks[coin_i] - market_cap_ranks[coin_j])
                if rank_diff <= 2:
                    domain_weight += 0.2
                elif rank_diff <= 4:
                    domain_weight += 0.1

            # 3. BTC影响力加权（所有币种都与BTC有关联）
            if coin_i == 'BTC' or coin_j == 'BTC':
                domain_weight += 0.2

            # 4. ETH生态加权
            if coin_i == 'ETH' or coin_j == 'ETH':
                domain_weight += 0.15

            # 综合权重
            final_weight = 0.6 * base_weight + 0.4 * domain_weight

            if final_weight > 0.2:
                edges.extend([[i, j], [j, i]])
                edge_weights.extend([final_weight, final_weight])

    if not edges:
        # 创建基于市值排名的连接
        sorted_coins = sorted([(coin, market_cap_ranks.get(coin, 999)) for coin in coin_names],
                            key=lambda x: x[1])
        for i in range(len(sorted_coins) - 1):
            idx_i = coin_names.index(sorted_coins[i][0])
            idx_j = coin_names.index(sorted_coins[i+1][0])
            edges.extend([[idx_i, idx_j], [idx_j, idx_i]])
            edge_weights.extend([0.3, 0.3])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    return edge_index, edge_weights

def _generate_attention_graph(input_series, top_k=3, use_returns=True):
    """
    基于注意力机制的图构建：为每个节点选择最相关的k个邻居

    Args:
        top_k: 每个节点的最大邻居数
        use_returns: 是否使用收益率而不是价格
    """
    if use_returns:
        data = input_series.pct_change().fillna(0)
    else:
        data = input_series

    n_nodes = len(data.columns)

    # 计算注意力权重矩阵
    attention_matrix = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                # 使用多种相似性度量的组合
                # 1. 皮尔逊相关系数
                corr = data.iloc[:, i].corr(data.iloc[:, j])

                # 2. 互信息（简化版本：基于分位数）
                def mutual_info_simple(x, y, bins=10):
                    x_binned = pd.cut(x, bins=bins, labels=False)
                    y_binned = pd.cut(y, bins=bins, labels=False)
                    contingency = pd.crosstab(x_binned, y_binned)
                    return contingency.values

                try:
                    mi_matrix = mutual_info_simple(data.iloc[:, i], data.iloc[:, j])
                    mi_score = np.sum(mi_matrix * np.log(mi_matrix + 1e-10)) / (mi_matrix.sum() + 1e-10)
                except:
                    mi_score = 0

                # 3. 动态时间规整距离（简化版本）
                def dtw_distance_simple(x, y, window=10):
                    x_vals = x.values[-window:]
                    y_vals = y.values[-window:]
                    return 1 / (1 + np.mean((x_vals - y_vals) ** 2))

                dtw_sim = dtw_distance_simple(data.iloc[:, i], data.iloc[:, j])

                # 综合注意力分数
                attention_score = 0.5 * abs(corr) + 0.3 * abs(mi_score) + 0.2 * dtw_sim
                attention_matrix[i, j] = attention_score

    # 为每个节点选择top-k邻居
    edges = []
    edge_weights = []

    for i in range(n_nodes):
        # 获取节点i的所有邻居的注意力分数
        neighbors_scores = attention_matrix[i, :]

        # 选择top-k邻居（排除自己）
        top_k_indices = np.argsort(neighbors_scores)[-top_k-1:-1]  # 排除最后一个（自己）

        for j in top_k_indices:
            if neighbors_scores[j] > 0.1:  # 最小阈值
                edges.append([i, j])
                edge_weights.append(neighbors_scores[j])

    # 确保图是无向的
    undirected_edges = []
    undirected_weights = []
    edge_set = set()

    for i, (edge, weight) in enumerate(zip(edges, edge_weights)):
        edge_tuple = tuple(sorted([edge[0], edge[1]]))
        if edge_tuple not in edge_set:
            edge_set.add(edge_tuple)
            undirected_edges.extend([[edge[0], edge[1]], [edge[1], edge[0]]])
            undirected_weights.extend([weight, weight])

    if not undirected_edges:
        # 创建最小连通图
        for i in range(n_nodes - 1):
            undirected_edges.extend([[i, i+1], [i+1, i]])
            undirected_weights.extend([0.1, 0.1])

    edge_index = torch.tensor(undirected_edges, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(undirected_weights, dtype=torch.float32)

    return edge_index, edge_weights

def analyze_graph_properties(edge_index, edge_weights=None, num_nodes=None):
    """
    分析图的属性
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    num_edges = edge_index.shape[1] // 2  # 无向图
    density = num_edges / (num_nodes * (num_nodes - 1) / 2)

    # 计算度分布
    degrees = torch.zeros(num_nodes)
    for i in range(num_nodes):
        degrees[i] = (edge_index[0] == i).sum().item()

    avg_degree = degrees.mean().item()
    max_degree = degrees.max().item()
    min_degree = degrees.min().item()

    properties = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'degree_std': degrees.std().item()
    }

    if edge_weights is not None:
        properties['avg_edge_weight'] = edge_weights.mean().item()
        properties['edge_weight_std'] = edge_weights.std().item()

    return properties

def generate_sliding_features_labels(input_series, input_dim, task='classification', norm_type=None):
    num_nodes = input_series.shape[1]
    coin_features_list = []
    coin_labels_list = []
    scaler = None
    # 对每个币种分别归一化/标准化
    for i in range(num_nodes):
        coin_data = input_series.iloc[:, i].values.reshape(-1, 1)
        if norm_type == 'standard':
            scaler = StandardScaler()
            coin_data = scaler.fit_transform(coin_data).flatten()
        elif norm_type == 'minmax':
            scaler = MinMaxScaler()
            coin_data = scaler.fit_transform(coin_data).flatten()
        else:
            coin_data = coin_data.flatten()
        coin_tensor = torch.tensor(coin_data).float()
        coin_features = []
        coin_labels = []
        for j in range(len(coin_tensor) - input_dim):
            features = coin_tensor[j:j + input_dim]
            label = coin_tensor[j + input_dim]
            if task == 'classification':
                label = 0 if label <= 0 else 1
            coin_features.append(features)
            coin_labels.append(label)
        coin_features_list.append(coin_features)
        coin_labels_list.append(coin_labels)
    return coin_features_list, coin_labels_list

def load_gnn_data(file_path, input_dim, threshold=0.6, task='classification', norm_type=None):
    data = pd.read_csv(file_path)
    input_series = data.drop('date', axis=1)
    edge_index = generate_edge_index(input_series, threshold)
    coin_features_list, coin_labels_list = generate_sliding_features_labels(input_series, input_dim, task, norm_type)
    return coin_features_list, coin_labels_list, edge_index

def create_gnn_dataloaders(
    coin_features_list, 
    coin_labels_list, 
    batch_size=32, 
    test_size=0.2, 
    seed=42, 
    task='classification'
):
    coin_train_loaders = []
    coin_test_loaders = []

    for coin_features, coin_labels in zip(coin_features_list, coin_labels_list):
        coin_features_tensor = torch.stack([
            f.clone().detach() if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.float32)
            for f in coin_features
        ])
        if task == 'classification':
            coin_labels_tensor = torch.tensor(coin_labels, dtype=torch.long).view(-1, 1)
        else:
            coin_labels_tensor = torch.tensor(coin_labels, dtype=torch.float32).view(-1, 1)

        train_input, test_input, train_labels, test_labels = train_test_split(
            coin_features_tensor.numpy(),
            coin_labels_tensor.numpy(),
            test_size=test_size,
            random_state=seed
        )

        train_input = torch.tensor(train_input).float()
        test_input = torch.tensor(test_input).float()
        if task == 'classification':
            train_labels = torch.tensor(train_labels).long()
            test_labels = torch.tensor(test_labels).long()
        else:
            train_labels = torch.tensor(train_labels).float()
            test_labels = torch.tensor(test_labels).float()

        train_dataset = TensorDataset(train_input, train_labels)
        test_dataset = TensorDataset(test_input, test_labels)

        coin_train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        coin_test_loaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

    return coin_train_loaders, coin_test_loaders

# 用法示例
if __name__ == "__main__":
    file_path = 'Project1/datafiles/1H.csv'
    input_dim = 30
    batch_size = 32
    test_size = 0.2
    threshold = 0.6

    print('--- 分类任务示例（标准化） ---')
    coin_features_list, coin_labels_list, edge_index = load_gnn_data(
        file_path=file_path,
        input_dim=input_dim,
        threshold=threshold,
        task="classification",
        norm_type='standard'
    )
    coin_train_loaders, coin_test_loaders = create_gnn_dataloaders(
        coin_features_list,
        coin_labels_list,
        batch_size=batch_size,
        test_size=test_size,
        task='classification'
    )
    print('edge_index:', edge_index.shape)
    for i, loader in enumerate(coin_train_loaders):
        for batch_x, batch_y in loader:
            print(f'币种{i} 训练batch特征shape:', batch_x.shape, '标签shape:', batch_y.shape)
            print('标签类型:', batch_y.dtype)
            break
        break

    print('--- 回归任务示例（归一化） ---')
    coin_features_list, coin_labels_list, edge_index = load_gnn_data(
        file_path=file_path,
        input_dim=input_dim,
        threshold=threshold,
        task='regression',
        norm_type='minmax'
    )
    coin_train_loaders, coin_test_loaders = create_gnn_dataloaders(
        coin_features_list,
        coin_labels_list,
        batch_size=batch_size,
        test_size=test_size,
        task='regression'
    )
    print('edge_index:', edge_index.shape)
    for i, loader in enumerate(coin_train_loaders):
        for batch_x, batch_y in loader:
            print(f'币种{i} 训练batch特征shape:', batch_x.shape, '标签shape:', batch_y.shape)
            print('标签类型:', batch_y.dtype)
            break
        break
