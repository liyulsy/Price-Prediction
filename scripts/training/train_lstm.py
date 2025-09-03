# === 导入必要的库 ===
import torch                    # PyTorch深度学习框架
import torch.nn as nn          # 神经网络模块
import torch.optim as optim    # 优化器模块
from torch.utils.data import DataLoader, Subset  # 数据加载和划分工具
import pandas as pd            # 数据处理库
from tqdm import tqdm         # 进度条显示
import os                     # 操作系统接口
import sys                    # 系统相关参数和函数
import numpy as np            # 数值计算库
import random                 # 随机数生成
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # 数据预处理
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error  # 评估指标
import csv                    # CSV文件处理
from datetime import datetime # 日期时间处理

# === 项目路径配置 ===
# 将项目根目录添加到Python路径中，确保能够导入自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# === 模型和数据集导入 ===
# 导入LSTM-GCN统一模型：结合长短期记忆网络和图卷积网络
from models.MixModel.unified_lstm_gcn import UnifiedLstmGnn
# 导入数据集处理类：处理加密货币价格和新闻数据
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
# 导入图构建工具：用于构建加密货币之间的关系图
from dataloader.gnn_loader import generate_edge_index, generate_advanced_edge_index, analyze_graph_properties

# === 设备配置 ===
# 自动检测并选择最佳计算设备：优先使用GPU，如果没有GPU则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 主要功能开关 ===
# 这些开关控制模型的核心功能，可以根据实验需求灵活调整
# 预测目标配置：
# - 'price': 预测绝对价格 (仅回归)
# - 'diff': 预测价格差分 (仅分类)
# - 'return': 预测价格变化率 (仅分类)
PREDICTION_TARGET = 'diff'

# 任务类型自动确定
TASK_TYPE = 'regression' if PREDICTION_TARGET == 'price' else 'classification'
USE_GCN = False                 # 是否使用图卷积网络：True=启用GCN, False=仅使用LSTM
USE_NEWS_FEATURES = True      # 是否使用新闻特征：True=包含新闻数据, False=仅使用价格数据

# === 图构建配置 ===
# 图构建方法选择：定义如何构建加密货币之间的关系图
# 基于实验结果，原始相关性方法在大多数情况下表现最佳
GRAPH_METHOD = 'dynamic'  # 图构建方法选择
# 可选方法：
#   'original': 基于皮尔逊相关系数的简单图构建
#   'multi_layer': 结合相关性、波动性、趋势的多层图
#   'dynamic': 基于滑动窗口的动态时变图
#   'domain_knowledge': 基于加密货币领域知识的图
#   'attention_based': 基于注意力机制的图

# 不同图构建方法的参数配置
GRAPH_PARAMS = {
    'original': {'threshold': 0.6},  # 原始方法：相关性阈值0.6
    'multi_layer': {  # 多层图方法：使用多种指标
        'correlation_threshold': 0.3,   # 相关性阈值
        'volatility_threshold': 0.5,    # 波动性阈值
        'trend_threshold': 0.4          # 趋势阈值
    },
    'dynamic': {  # 动态图方法：时间窗口滑动
        'window_size': 168,  # 窗口大小：168小时(7天)
        'overlap': 24        # 重叠大小：24小时(1天)
    },
    'domain_knowledge': {  # 领域知识方法：基于币种类型
        'coin_names': ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
    },
    'attention_based': {  # 注意力方法：学习重要连接
        'top_k': 3,          # 保留前3个最重要的连接
        'use_returns': True  # 使用收益率数据
    }
}

# === GCN架构配置 ===
# GCN配置：选择图卷积网络的架构类型
GCN_CONFIG = 'improved_light'  # GCN架构选择
# 可选配置：
#   'basic': 基础2层GCN
#   'improved_light': 轻量级改进GCN（3层，ReLU激活，残差连接，批归一化）
#   'improved_gelu': GELU激活改进GCN（3层，GELU激活，残差连接，批归一化）
#   'gat_attention': 图注意力网络（GAT，2层，4个注意力头）
#   'adaptive': 自适应GCN（3层，自适应dropout和激活函数）

# --- Data & Cache Paths ---
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
NEWS_FEATURES_FOLDER = 'scripts/analysis/crypto_new_analyzer/features'
CACHE_DIR = "experiments/cache"
BEST_MODEL_NAME = "best_lstm_model.pt"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 60  # LSTM适合的序列长度
THRESHOLD = 0.6
NORM_TYPE = None  # LSTM使用标准化
TIME_ENCODING_ENABLED_IN_DATASET = True
TIME_FREQ_IN_DATASET = 'h'

# --- LSTM Model Parameters ---
NEWS_PROCESSED_DIM = 32
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
GCN_HIDDEN_DIM = 128
GCN_OUTPUT_DIM = 64
MLP_HIDDEN_DIM = 256
NUM_CLASSES = 1 if TASK_TYPE == 'regression' else 2

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 50  # 增加训练轮数
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
FORCE_RECOMPUTE_NEWS = False
RANDOM_SEED = 42
# 早停参数
EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 1e-6

def set_random_seeds(seed=42):
    """设置所有随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# --- Dynamic File Path ---
# 根据任务类型创建子目录
task_dir = "classification" if TASK_TYPE == "classification" else "regression"
model_save_dir = os.path.join(CACHE_DIR, task_dir)

# 确保保存目录存在
os.makedirs(model_save_dir, exist_ok=True)

model_variant = ['LSTM', TASK_TYPE]
model_variant.append("with_gcn" if USE_GCN else "no_gcn")
model_variant.append("with_news" if USE_NEWS_FEATURES else "no_news")
model_variant_str = "_".join(model_variant)
BEST_MODEL_PATH = os.path.join(model_save_dir, f"{model_variant_str}_{BEST_MODEL_NAME}")
print(f"--- Configuration: {model_variant_str} ---")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

def save_test_predictions(all_preds, all_targets, coin_names, model_name, test_metrics=None):
    """保存测试集的预测值和真实值到CSV文件"""
    base_save_dir = "experiments/cache/test_predictions"
    model_save_dir = os.path.join(base_save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    predictions_file = os.path.join(model_save_dir, "test_predictions.csv")
    with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_idx', 'coin', 'true_value', 'predicted_value', 'absolute_error', 'percentage_error'])

        for sample_idx in range(len(all_preds)):
            for coin_idx, coin_name in enumerate(coin_names):
                true_val = all_targets[sample_idx, coin_idx]
                pred_val = all_preds[sample_idx, coin_idx]
                abs_error = abs(true_val - pred_val)
                pct_error = (abs_error / abs(true_val)) * 100 if abs(true_val) > 1e-8 else float('inf')
                writer.writerow([sample_idx, coin_name, true_val, pred_val, abs_error, pct_error])

    statistics_file = os.path.join(model_save_dir, "test_statistics.csv")
    with open(statistics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['coin', 'mean_true', 'mean_pred', 'std_true', 'std_pred',
                        'min_true', 'min_pred', 'max_true', 'max_pred', 'mae', 'mape'])

        for coin_idx, coin_name in enumerate(coin_names):
            true_vals = all_targets[:, coin_idx]
            pred_vals = all_preds[:, coin_idx]
            mae = mean_absolute_error(true_vals, pred_vals)
            mape = np.mean(np.abs((true_vals - pred_vals) / np.where(np.abs(true_vals) > 1e-8, true_vals, 1e-8))) * 100

            writer.writerow([
                coin_name,
                np.mean(true_vals), np.mean(pred_vals),
                np.std(true_vals), np.std(pred_vals),
                np.min(true_vals), np.min(pred_vals),
                np.max(true_vals), np.max(pred_vals),
                mae, mape
            ])

    # === 保存格式化的测试结果 (TXT格式) ===
    if test_metrics:
        results_txt_file = os.path.join(model_save_dir, "test_results.txt")

        with open(results_txt_file, 'w', encoding='utf-8') as f:
            f.write("🎉 最终测试结果\n")
            f.write("="*60 + "\n")
            f.write("📊 整体指标:\n")

            # 写入整体指标
            for name, value in test_metrics.items():
                if not isinstance(value, dict):  # 跳过嵌套字典
                    if isinstance(value, (int, float)):
                        # 为不同指标添加中文注释
                        if name.upper() == 'LOSS':
                            comment = "# 测试损失 - 模型在测试集上的损失值"
                        elif name.upper() == 'MAE':
                            comment = "# 平均绝对误差 - 预测值与真实值的平均绝对差"
                        elif name.upper() == 'RD':
                            comment = "# 相对偏差 - |1-预测值总和/真实值总和|"
                        elif name.upper() == 'MSE':
                            comment = "# 均方误差 - 预测值与真实值差的平方的平均"
                        elif name.upper() == 'RMSE':
                            comment = "# 均方根误差 - MSE的平方根"
                        elif name.upper() == 'R2':
                            comment = "# 决定系数 - 模型解释数据变异性的比例(越接近1越好)"
                        elif name.upper() == 'MAPE':
                            comment = "# 平均绝对百分比误差 - 相对误差的百分比"
                        elif 'NORMALIZED' in name.upper():
                            comment = f"# 归一化{name.split('_')[-1]} - 消除币种价格尺度影响的{name.split('_')[-1]}"
                        else:
                            comment = ""

                        if name.upper() == 'RD':
                            f.write(f"    - RD: {value:.4f}  {comment}\n")
                        else:
                            f.write(f"    - {name.upper()}: {value:.4f}  {comment}\n")
                    else:
                        f.write(f"    - {name.upper()}: {value}\n")

            f.write("\n📈 各币种详细指标:\n")

            # 写入各币种详细指标
            if 'per_coin_metrics' in test_metrics:
                for coin_name, coin_metrics in test_metrics['per_coin_metrics'].items():
                    f.write(f"  🪙 {coin_name}:\n")
                    for metric_name, metric_value in coin_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            # 为不同指标添加中文注释
                            if metric_name.lower() == 'normalized_mae':
                                comment = f"# {coin_name}的归一化MAE"
                            elif metric_name.lower() == 'normalized_mse':
                                comment = f"# {coin_name}的归一化MSE"
                            elif metric_name.lower() == 'normalized_rmse':
                                comment = f"# {coin_name}的归一化RMSE"
                            elif metric_name.lower() == 'mape':
                                comment = f"# {coin_name}的平均绝对百分比误差"
                            elif metric_name.lower() == 'r2':
                                comment = f"# {coin_name}的决定系数"
                            elif metric_name.lower() == 'mae':
                                comment = f"# {coin_name}的平均绝对误差"
                            elif metric_name.lower() == 'mse':
                                comment = f"# {coin_name}的均方误差"
                            elif metric_name.lower() == 'rmse':
                                comment = f"# {coin_name}的均方根误差"
                            else:
                                comment = f"# {coin_name}的{metric_name}"
                            f.write(f"    - {metric_name.upper()}: {metric_value:.4f}  {comment}\n")
                        else:
                            f.write(f"    - {metric_name.upper()}: {metric_value}\n")

            f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"测试集预测结果已保存到 {model_save_dir}:")
    print(f"  详细结果: test_predictions.csv")
    print(f"  统计信息: test_statistics.csv")
    if test_metrics:
        print(f"  格式化结果: test_results.txt")
    return predictions_file, statistics_file

def evaluate_model(model, data_loader, criterion, edge_index, edge_weights, device, task_type, scaler=None):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating"):
            price_seq = batch_data['price_seq'].to(device)
            target_data = batch_data['target_price'].to(device)
            x_mark_enc = batch_data.get('price_seq_mark')
            if x_mark_enc is not None: x_mark_enc = x_mark_enc.to(device)
            news_features = batch_data.get('news_features')
            if news_features is not None: news_features = news_features.to(device)
            
            outputs = model(price_seq, edge_index=edge_index, edge_weight=edge_weights, news_features=news_features)
            
            if task_type == 'classification':
                # === 分类任务：预测价格涨跌方向 ===
                if PREDICTION_TARGET in ('diff', 'return'):
                    # diff/return：target 已是差分/收益率，直接判断正负
                    targets = (target_data > 0).long()
                else:
                    raise ValueError(f"PREDICTION_TARGET '{PREDICTION_TARGET}' not supported for classification")

                # 确保输出和目标的形状匹配
                if len(outputs.shape) == 3:  # [batch_size, num_nodes, num_classes]
                    outputs_flat = outputs.view(-1, NUM_CLASSES)
                    targets_flat = targets.view(-1)
                elif len(outputs.shape) == 2:  # [batch_size * num_nodes, num_classes]
                    outputs_flat = outputs
                    targets_flat = targets.view(-1)
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                loss = criterion(outputs_flat, targets_flat)
                preds = torch.argmax(outputs, dim=-1)
            else: # regression
                if PREDICTION_TARGET == 'price':
                    targets = target_data
                else:
                    raise ValueError(f"PREDICTION_TARGET '{PREDICTION_TARGET}' not supported for regression")
                loss = criterion(outputs, targets)
                preds = outputs
                
            total_loss += loss.item() * price_seq.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = {'loss': avg_loss}

    # === 计算指标：根据任务类型计算不同的评估指标 ===
    if task_type == 'classification':
        # === 分类任务指标计算 ===
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        # 整体分类指标
        overall_accuracy = accuracy_score(all_targets.flatten(), all_preds.flatten())
        overall_precision = precision_score(all_targets.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        overall_recall = recall_score(all_targets.flatten(), all_preds.flatten(), average='weighted', zero_division=0)
        overall_f1 = f1_score(all_targets.flatten(), all_preds.flatten(), average='weighted', zero_division=0)

        # 按类别分类指标
        precision_per_class = precision_score(all_targets.flatten(), all_preds.flatten(), average=None, zero_division=0)
        recall_per_class = recall_score(all_targets.flatten(), all_preds.flatten(), average=None, zero_division=0)
        f1_per_class = f1_score(all_targets.flatten(), all_preds.flatten(), average=None, zero_division=0)

        # 每个币种的分类指标
        per_coin_metrics = {}
        coin_accuracies = []
        coin_precisions = []
        coin_recalls = []
        coin_f1s = []

        for i, coin_name in enumerate(COIN_NAMES):
            coin_targets = all_targets[:, i]
            coin_preds = all_preds[:, i]

            coin_accuracy = accuracy_score(coin_targets, coin_preds)
            coin_precision = precision_score(coin_targets, coin_preds, average='weighted', zero_division=0)
            coin_recall = recall_score(coin_targets, coin_preds, average='weighted', zero_division=0)
            coin_f1 = f1_score(coin_targets, coin_preds, average='weighted', zero_division=0)

            coin_accuracies.append(coin_accuracy)
            coin_precisions.append(coin_precision)
            coin_recalls.append(coin_recall)
            coin_f1s.append(coin_f1)

            per_coin_metrics[coin_name] = {
                'accuracy': coin_accuracy,
                'precision': coin_precision,
                'recall': coin_recall,
                'f1_score': coin_f1
            }

        # 计算混淆矩阵，确保是2x2
        conf_matrix = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=[0, 1])

        # 更新指标字典
        metrics.update({
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'avg_accuracy': np.mean(coin_accuracies),
            'avg_precision': np.mean(coin_precisions),
            'avg_recall': np.mean(coin_recalls),
            'avg_f1_score': np.mean(coin_f1s),
            'precision_class_0': precision_per_class[0] if len(precision_per_class) > 0 else 0,
            'precision_class_1': precision_per_class[1] if len(precision_per_class) > 1 else 0,
            'recall_class_0': recall_per_class[0] if len(recall_per_class) > 0 else 0,
            'recall_class_1': recall_per_class[1] if len(recall_per_class) > 1 else 0,
            'f1_class_0': f1_per_class[0] if len(f1_per_class) > 0 else 0,
            'f1_class_1': f1_per_class[1] if len(f1_per_class) > 1 else 0,
            'confusion_matrix': conf_matrix.tolist()
        })

        metrics['per_coin_metrics'] = per_coin_metrics

    elif task_type == 'regression':
        # Denormalize for metrics calculation
        if PREDICTION_TARGET == 'price':
            if scaler:
                num_coins = all_preds.shape[1]
                original_preds = scaler.inverse_transform(all_preds.reshape(-1, num_coins))
                original_targets = scaler.inverse_transform(all_targets.reshape(-1, num_coins))
            else:
                original_preds = all_preds
                original_targets = all_targets
        else:
            original_preds = all_preds
            original_targets = all_targets

        # Calculate per-coin means for normalization
        coin_means = np.mean(original_targets, axis=0)
        coin_maes, coin_mses, coin_mapes, coin_r2s = [], [], [], []
        
        # Per-coin regression metrics
        per_coin_metrics = {}
        for i, coin_name in enumerate(COIN_NAMES):
            coin_targets = original_targets[:, i]
            coin_preds = original_preds[:, i]
            coin_mean = coin_means[i]
            
            # Calculate normalized errors
            norm_targets = coin_targets / coin_mean
            norm_preds = coin_preds / coin_mean
            
            # Calculate metrics
            mae = mean_absolute_error(norm_targets, norm_preds)
            mse = mean_squared_error(norm_targets, norm_preds)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(coin_targets, coin_preds)
            r2 = r2_score(coin_targets, coin_preds)
            
            coin_maes.append(mae)
            coin_mses.append(mse)
            coin_mapes.append(mape)
            coin_r2s.append(r2)
            
            per_coin_metrics[coin_name] = {
                'normalized_mae': mae,
                'normalized_mse': mse,
                'normalized_rmse': rmse,
                'mape': mape,
                'r2': r2,
                'mae': mean_absolute_error(coin_targets, coin_preds),
                'mse': mean_squared_error(coin_targets, coin_preds),
                'rmse': np.sqrt(mean_squared_error(coin_targets, coin_preds))
            }
        
        # Calculate RD: |1 - sum of all predicted values / sum of all true values|
        total_true_sum = np.sum(original_targets)
        total_pred_sum = np.sum(original_preds)
        rd = abs(1 - total_pred_sum / total_true_sum) if total_pred_sum != 0 else float('inf')

        # Calculate overall metrics
        metrics.update({
            'mae': mean_absolute_error(original_targets, original_preds),
            'rd': rd,
            'mse': mean_squared_error(original_targets, original_preds),
            'rmse': np.sqrt(mean_squared_error(original_targets, original_preds)),
            'r2': r2_score(original_targets, original_preds),
            'mape': mean_absolute_percentage_error(original_targets, original_preds),
            'normalized_mae': np.mean(coin_maes),
            'normalized_mse': np.mean(coin_mses),
            'normalized_rmse': np.sqrt(np.mean(coin_mses)),
            'avg_mape': np.mean(coin_mapes),
            'avg_r2': np.mean(coin_r2s),
            'median_mape': np.median(coin_mapes),
            'worst_mape': np.max(coin_mapes),
            'best_mape': np.min(coin_mapes)
        })
        
        metrics['per_coin_metrics'] = per_coin_metrics

    return metrics, all_preds, all_targets

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Load Data
    print("📊 加载价格数据...")
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    # 确保时间索引是升序排列（从早到晚）
    if not price_df_full.index.is_monotonic_increasing:
        print(f"⚠️ 价格数据时间索引不是升序，正在排序...")
        price_df_full = price_df_full.sort_index()
        print(f"✅ 价格数据已按时间升序排列")

    # 2. Generate Edge Index
    if USE_GCN:
        print(f"🔗 构建图结构，使用方法: {GRAPH_METHOD}")

        if GRAPH_METHOD == 'original':
            # 原始方法：基于相关性构建图，现在也支持边权重
            edge_index, edge_weights = generate_edge_index(
                price_df_full,
                return_weights=True,  # 请求返回边权重
                **GRAPH_PARAMS[GRAPH_METHOD]
            )
            # 将图数据移动到指定设备（GPU或CPU）
            edge_index = edge_index.to(DEVICE)
            edge_weights = edge_weights.to(DEVICE) if edge_weights is not None else None
        else:
            # 高级方法：使用更复杂的图构建算法
            edge_index, edge_weights = generate_advanced_edge_index(
                price_df_full,                    # 价格数据
                method=GRAPH_METHOD,              # 图构建方法
                **GRAPH_PARAMS[GRAPH_METHOD]      # 方法特定参数
            )
            # 将图数据移动到指定设备（GPU或CPU）
            edge_index = edge_index.to(DEVICE)
            edge_weights = edge_weights.to(DEVICE) if edge_weights is not None else None

        # 分析并打印图的属性（连接数、密度等）
        graph_properties = analyze_graph_properties(edge_index, edge_weights, len(COIN_NAMES))
        print(f"📈 图属性分析:")
        for key, value in graph_properties.items():
            # 格式化输出：浮点数保留4位小数，其他直接输出
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        # 不使用GCN时，图相关变量设为None
        print("🚫 未启用GCN，跳过图构建")
        edge_index = None
        edge_weights = None

    # 3. Normalize Data (target-aware)
    print(f"🔄 数据预处理: target={PREDICTION_TARGET}, norm={NORM_TYPE}")

    if PREDICTION_TARGET == 'diff':
        df_to_scale = price_df_full.diff().dropna()
    elif PREDICTION_TARGET == 'return':
        df_to_scale = price_df_full.pct_change().dropna()
    else:
        df_to_scale = price_df_full

    # 选择归一化器（差分/变化率数据需要特殊处理）
    if NORM_TYPE == 'none':
        scaler = None
    elif PREDICTION_TARGET in ('diff', 'return'):
        # 对于差分/变化率，归一化可能破坏正负分布
        print(f"⚠️  差分/变化率数据使用归一化，请注意正负样本分布")
        if NORM_TYPE == 'standard':
            scaler = StandardScaler()
        elif NORM_TYPE == 'minmax':
            print(f"⚠️  MinMax归一化可能不适合差分数据，建议使用 'standard' 或 'none'")
            scaler = MinMaxScaler()
        else:
            scaler = None
    else:
        # 价格数据正常归一化
        scaler = StandardScaler() if NORM_TYPE == 'standard' else MinMaxScaler() if NORM_TYPE == 'minmax' else None

    # 执行归一化（如选择了归一化器）
    if scaler:
        price_df_values = scaler.fit_transform(df_to_scale)
        price_df = pd.DataFrame(price_df_values, columns=df_to_scale.columns, index=df_to_scale.index)
        print(f"✅ 数据归一化完成，方法: {NORM_TYPE}")

        # 检查差分数据归一化后的分布
        if PREDICTION_TARGET in ('diff', 'return'):
            pos_ratio = (price_df > 0).sum().sum() / price_df.size
            print(f"🔍 归一化后正值比例: {pos_ratio:.1%}")
            if pos_ratio < 0.1 or pos_ratio > 0.9:
                print(f"⚠️  正负样本严重不平衡！建议使用 NORM_TYPE='none'")
    else:
        price_df = df_to_scale
        print(f"✅ 跳过归一化，使用原始数据")

    # 4. Create Dataset and DataLoaders
    print("📰 加载新闻数据..." if USE_NEWS_FEATURES else "🚫 跳过新闻数据加载")
    news_data_dict = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) if USE_NEWS_FEATURES else None

    # 调试：检查新闻数据加载情况
    if USE_NEWS_FEATURES:
        if news_data_dict:
            print(f"🔍 新闻数据加载检查:")
            for coin, data in news_data_dict.items():
                if data and 'news' in data:
                    print(f"  {coin}: {len(data['news'])} 条新闻")
                else:
                    print(f"  {coin}: 无新闻数据")
        else:
            print(f"❌ 新闻数据字典为空！")
            print(f"📁 检查新闻文件夹: {NEWS_FEATURES_FOLDER}")
            if os.path.exists(NEWS_FEATURES_FOLDER):
                files = os.listdir(NEWS_FEATURES_FOLDER)
                print(f"  文件夹内容: {files}")
            else:
                print(f"  文件夹不存在！")

        # 对于diff/return，提示可能的索引不匹配问题
        if PREDICTION_TARGET in ('diff', 'return'):
            print(f"🔄 差分/变化率模式：将尝试自动对齐现有新闻特征")

    if USE_NEWS_FEATURES:
        processed_news_path = os.path.join(CACHE_DIR, "news_features", "all_processed_news_feature_new10days.pt")
    else:
        processed_news_path = None
        FORCE_RECOMPUTE_NEWS = False

    print(f"🔄 创建数据集...")
    print(f"  价格数据形状: {price_df.shape}")
    print(f"  价格数据时间范围: {price_df.index[0]} 到 {price_df.index[-1]}")
    print(f"  价格数据时间顺序: {'升序' if price_df.index[0] < price_df.index[-1] else '降序'}")
    print(f"  新闻数据: {'已加载' if news_data_dict else '未加载'}")
    print(f"  强制重新计算新闻: {FORCE_RECOMPUTE_NEWS}")

    # 检查价格数据的时间顺序
    if len(price_df.index) > 1:
        time_diff = (price_df.index[1] - price_df.index[0]).total_seconds()
        if time_diff > 0:
            print(f"  ✅ 价格数据时间顺序正确：从早到晚")
        else:
            print(f"  ⚠️ 价格数据时间顺序：从晚到早，需要检查是否影响新闻特征对齐")
            print(f"  💡 建议：确保新闻特征也按相同顺序排列")

    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        news_data_dict=news_data_dict,
        seq_len=PRICE_SEQ_LEN,
        processed_news_features_path=processed_news_path,
        force_recompute_news=FORCE_RECOMPUTE_NEWS,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET,
        time_freq=TIME_FREQ_IN_DATASET,
    )

    # 使用时间序列正确的划分方式，避免数据泄露
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size

    print(f"📊 时间序列数据划分:")
    print(f"  训练集: 0 到 {train_size-1} ({train_size} 样本)")
    print(f"  验证集: {train_size} 到 {train_size+val_size-1} ({val_size} 样本)")
    print(f"  测试集: {train_size+val_size} 到 {total_size-1} ({test_size} 样本)")

    # 按时间顺序划分，避免随机划分导致的数据泄露
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize LSTM Model
    print(f"🚀 初始化LSTM模型，GCN配置: {GCN_CONFIG}")
    model = UnifiedLstmGnn(
        seq_len=PRICE_SEQ_LEN,  # 修正：使用 seq_len 而不是 price_seq_len
        num_nodes=dataset.num_coins,
        input_dim=1,  # 价格特征维度
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_out_dim=32,  # LSTM输出维度，使用默认值
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        final_mlp_hidden_dim=MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        # Optional components switches
        task_type=TASK_TYPE,
        use_gcn=USE_GCN,
        gcn_config=GCN_CONFIG,  # GCN配置选择
        news_feature_dim=dataset.news_feature_dim if USE_NEWS_FEATURES else None,
        news_processed_dim=NEWS_PROCESSED_DIM
    ).to(DEVICE)

    print(model)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 6. Setup Loss, Optimizer
    criterion = nn.CrossEntropyLoss() if TASK_TYPE == 'classification' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, factor=0.8, min_lr=1e-7
    )
    print(f"📈 学习率调度器配置: patience=10, factor=0.8, min_lr=1e-7")

    # 7. Training Loop
    if TASK_TYPE == 'classification':
        best_val_metric = float('-inf')   # 分类任务：F1分数越大越好（使用负值，初始化为负无穷大）
    else:
        best_val_metric = float('inf')    # 回归任务：损失越小越好（初始化为正无穷大）
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")
        for batch_data in train_pbar:
            price_seq = batch_data['price_seq'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)
            x_mark_enc = batch_data.get('price_seq_mark')
            if x_mark_enc is not None: x_mark_enc = x_mark_enc.to(DEVICE)
            news_features = batch_data.get('news_features')
            if news_features is not None: news_features = news_features.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(price_seq, edge_index=edge_index, edge_weight=edge_weights, news_features=news_features)

            if TASK_TYPE == 'classification':
                # === 分类任务：预测价格涨跌方向 ===
                if PREDICTION_TARGET in ('diff', 'return'):
                    # diff/return：target 已是差分/收益率
                    targets = (target_data > 0).long()
                else:
                    raise ValueError(f"PREDICTION_TARGET '{PREDICTION_TARGET}' not supported for classification")

                # 确保输出和目标的形状匹配
                if len(outputs.shape) == 3:  # [batch_size, num_nodes, num_classes]
                    outputs_flat = outputs.view(-1, NUM_CLASSES)
                    targets_flat = targets.view(-1)
                elif len(outputs.shape) == 2:  # [batch_size * num_nodes, num_classes]
                    outputs_flat = outputs
                    targets_flat = targets.view(-1)
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                loss = criterion(outputs_flat, targets_flat)
            else:
                if PREDICTION_TARGET == 'price':
                    targets = target_data
                else:
                    raise ValueError(f"PREDICTION_TARGET '{PREDICTION_TARGET}' not supported for regression")
                loss = criterion(outputs, targets)

            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item() * price_seq.size(0)

        avg_train_loss = epoch_loss / len(train_dataset)
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, edge_index, edge_weights, DEVICE, TASK_TYPE, scaler)
        
        # 获取用于学习率调度的验证指标
        if TASK_TYPE == 'classification':
            # 分类任务使用F1分数（越大越好，需要取负值用于早停）
            val_metric_for_scheduler = -val_metrics.get('f1', 0)  # 取负值，因为早停机制是基于"越小越好"
        else:
            # 回归任务使用损失（越小越好）
            val_metric_for_scheduler = val_metrics['loss']

        # 根据验证指标调整学习率
        if TASK_TYPE == 'classification':
            # 分类任务：使用F1分数（传入正值给调度器）
            scheduler.step(-val_metric_for_scheduler)
        else:
            # 回归任务：使用损失
            scheduler.step(val_metric_for_scheduler)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")

        # 检查学习率是否过小
        if current_lr < 1e-6:
            print(f"⚠️ 学习率过小 ({current_lr:.2e})，可能影响训练效果")
        print("--- Validation Metrics (Overall) ---")
        for name, value in val_metrics.items():
            if not isinstance(value, dict):
                if isinstance(value, (int, float)):
                    # 为不同指标添加中文注释
                    if name == 'accuracy':
                        print(f"  - {name.upper()}: {value:.4f}  # 整体准确率 - 预测正确的样本比例")
                    elif name == 'precision':
                        print(f"  - {name.upper()}: {value:.4f}  # 整体精确率 - 预测为涨的样本中实际上涨的比例")
                    elif name == 'recall':
                        print(f"  - {name.upper()}: {value:.4f}  # 整体召回率 - 实际上涨的样本中被正确预测的比例")
                    elif name == 'f1_score':
                        print(f"  - {name.upper()}: {value:.4f}  # 整体F1分数 - 精确率和召回率的调和平均")
                    elif name == 'avg_accuracy':
                        print(f"  - {name.upper()}: {value:.4f}  # 平均准确率 - 各币种准确率的平均值")
                    elif name == 'avg_precision':
                        print(f"  - {name.upper()}: {value:.4f}  # 平均精确率 - 各币种精确率的平均值")
                    elif name == 'avg_recall':
                        print(f"  - {name.upper()}: {value:.4f}  # 平均召回率 - 各币种召回率的平均值")
                    elif name == 'avg_f1_score':
                        print(f"  - {name.upper()}: {value:.4f}  # 平均F1分数 - 各币种F1分数的平均值")
                    elif name == 'precision_class_0':
                        print(f"  - {name.upper()}: {value:.4f}  # 下跌类精确率 - 预测下跌中实际下跌的比例")
                    elif name == 'precision_class_1':
                        print(f"  - {name.upper()}: {value:.4f}  # 上涨类精确率 - 预测上涨中实际上涨的比例")
                    elif name == 'recall_class_0':
                        print(f"  - {name.upper()}: {value:.4f}  # 下跌类召回率 - 实际下跌中被正确预测的比例")
                    elif name == 'recall_class_1':
                        print(f"  - {name.upper()}: {value:.4f}  # 上涨类召回率 - 实际上涨中被正确预测的比例")
                    elif name == 'f1_class_0':
                        print(f"  - {name.upper()}: {value:.4f}  # 下跌类F1分数 - 下跌类精确率和召回率的调和平均")
                    elif name == 'f1_class_1':
                        print(f"  - {name.upper()}: {value:.4f}  # 上涨类F1分数 - 上涨类精确率和召回率的调和平均")
                    elif name == 'loss':
                        print(f"  - {name.upper()}: {value:.4f}  # 验证损失 - 模型在验证集上的损失值")
                    else:
                        print(f"  - {name.upper()}: {value:.4f}")
                elif isinstance(value, list):
                    if name == 'confusion_matrix':
                        print(f"  - {name.upper()}: {value}  # 混淆矩阵 - [[真负例,假正例],[假负例,真正例]]")
                    else:
                        print(f"  - {name.upper()}: {value}")
                else:
                    print(f"  - {name.upper()}: {value}")

        # 早停和模型保存逻辑
        if val_metric_for_scheduler < best_val_metric - MIN_DELTA:
            best_val_metric = val_metric_for_scheduler
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            if TASK_TYPE == 'classification':
                print(f"🚀 保存新的最佳模型到 {BEST_MODEL_PATH} (验证F1: {-best_val_metric:.4f})")
            else:
                print(f"🚀 保存新的最佳模型到 {BEST_MODEL_PATH} (验证损失: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            if TASK_TYPE == 'classification':
                print(f"⏳ 连续 {patience_counter} 个epoch无改善 (最佳F1: {-best_val_metric:.4f})")
            else:
                print(f"⏳ 连续 {patience_counter} 个epoch无改善 (最佳损失: {best_val_metric:.4f})")

        # 早停检查
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs (patience: {EARLY_STOPPING_PATIENCE})")
            break

    # 8. Testing
    print("\n--- Starting Testing with Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("Warning: Best model not found. Testing with the last state.")

    test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, criterion, edge_index, edge_weights, DEVICE, TASK_TYPE, scaler)

    # 保存测试集预测结果
    if TASK_TYPE == 'regression':
        if PREDICTION_TARGET == 'price' and scaler:
            original_test_preds = scaler.inverse_transform(test_preds)
            original_test_targets = scaler.inverse_transform(test_targets)
        else:
            original_test_preds = test_preds
            original_test_targets = test_targets

        save_test_predictions(original_test_preds, original_test_targets, COIN_NAMES, model_variant_str, test_metrics)

    print(f"\n✅ Test Results:")
    print("  Overall:")
    for name, value in test_metrics.items():
        if not isinstance(value, dict):
            if isinstance(value, (int, float)):
                # 为不同指标添加中文注释
                if name == 'accuracy':
                    print(f"    - {name.upper()}: {value:.4f}  # 整体准确率 - 预测正确的样本比例")
                elif name == 'precision':
                    print(f"    - {name.upper()}: {value:.4f}  # 整体精确率 - 预测为涨的样本中实际上涨的比例")
                elif name == 'recall':
                    print(f"    - {name.upper()}: {value:.4f}  # 整体召回率 - 实际上涨的样本中被正确预测的比例")
                elif name == 'f1_score':
                    print(f"    - {name.upper()}: {value:.4f}  # 整体F1分数 - 精确率和召回率的调和平均")
                elif name == 'loss':
                    print(f"    - {name.upper()}: {value:.4f}  # 测试损失 - 模型在测试集上的损失值")
                elif name == 'mae':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均绝对误差 - 预测值与真实值的平均绝对差")
                elif name == 'mse':
                    print(f"    - {name.upper()}: {value:.4f}  # 均方误差 - 预测值与真实值差的平方的平均")
                elif name == 'rmse':
                    print(f"    - {name.upper()}: {value:.4f}  # 均方根误差 - MSE的平方根")
                elif name == 'r2':
                    print(f"    - {name.upper()}: {value:.4f}  # 决定系数 - 模型解释数据变异性的比例")
                else:
                    print(f"    - {name.upper()}: {value:.4f}")
            elif isinstance(value, list):
                if name == 'confusion_matrix':
                    print(f"    - {name.upper()}: {value}  # 混淆矩阵 - [[真负例,假正例],[假负例,真正例]]")
                else:
                    print(f"    - {name.upper()}: {value}")
            else:
                print(f"    - {name.upper()}: {value}")

    if 'per_coin_metrics' in test_metrics:
        print(f"\n  --- 各币种详细指标 (测试集) ---")
        for coin_name, coin_metrics in test_metrics['per_coin_metrics'].items():
            print(f"  🪙 {coin_name}:")
            for metric_name, value in coin_metrics.items():
                if isinstance(value, (int, float)):
                    # 为每个币种的指标添加注释
                    if metric_name == 'accuracy':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的预测准确率")
                    elif metric_name == 'precision':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的预测精确率")
                    elif metric_name == 'recall':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的预测召回率")
                    elif metric_name == 'f1_score':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的F1分数")
                    elif metric_name == 'mae':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的平均绝对误差")
                    elif metric_name == 'mse':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的均方误差")
                    elif metric_name == 'rmse':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的均方根误差")
                    elif metric_name == 'r2':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的决定系数")
                    else:
                        print(f"    - {metric_name.upper()}: {value:.4f}")
                else:
                    print(f"    - {metric_name.upper()}: {value}")
    print("\n--- Script Finished ---")
