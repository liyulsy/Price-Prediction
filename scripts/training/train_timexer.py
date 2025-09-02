import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import csv
from datetime import datetime
 
# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# --- Model and Dataset Imports ---
# 导入TimeXer模型：结合了时间序列预测和图神经网络的统一模型
from models.MixModel.unified_timexer_gcn import UnifiedTimexerGCN
# 导入数据集处理类：处理加密货币价格和新闻数据
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
# 导入图构建工具：用于构建加密货币之间的关系图
from dataloader.gnn_loader import generate_edge_index, generate_advanced_edge_index, analyze_graph_properties

# --- Configuration ---
# 设备配置：优先使用GPU，如果没有GPU则使用CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Master Switches ---
# 主要开关：控制模型的核心功能
# 预测目标配置：
# - 'price': 预测绝对价格 (仅回归)
# - 'diff': 预测价格差分 (仅分类)
# - 'return': 预测价格变化率 (仅分类)
PREDICTION_TARGET = 'diff'

# 任务类型自动确定
TASK_TYPE = 'regression' if PREDICTION_TARGET == 'price' else 'classification'
USE_GCN = True                 # 是否使用图卷积网络：True=启用GCN, False=不使用GCN
USE_NEWS_FEATURES = False       # 是否使用新闻特征：True=包含新闻数据, False=仅使用价格数据

# --- Graph Construction Configuration ---
# 图构建配置：定义如何构建加密货币之间的关系图
# 基于实验结果，原始方法表现最佳！
GRAPH_METHOD = 'original'  # 图构建方法选择
# 可选方法：'original'(基于相关性), 'multi_layer'(多层图), 'dynamic'(动态图),
#          'domain_knowledge'(领域知识), 'attention_based'(注意力机制)

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

# --- GCN Configuration ---
# GCN配置：选择图卷积网络的架构类型
GCN_CONFIG = 'improved_light'  # GCN架构选择
# 可选配置：'basic'(基础GCN), 'improved_light'(轻量改进), 'improved_gelu'(GELU激活),
#          'gat_attention'(图注意力), 'adaptive'(自适应GCN)

# --- Data & Cache Paths ---
# 数据和缓存路径配置
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'  # 价格数据文件路径
NEWS_FEATURES_FOLDER = 'scripts/analysis/crypto_new_analyzer/features'                # 新闻特征文件夹路径
CACHE_DIR = "experiments/caches"        # 缓存目录：存储模型和中间结果
BEST_MODEL_NAME = "best_timexer_model.pt"  # 最佳模型文件名

# --- Dataset Parameters ---
# 数据集参数配置
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']  # 要分析的加密货币列表
PRICE_SEQ_LEN = 90              # 价格序列长度：使用过去90个时间点的数据进行预测
THRESHOLD = 0.6                 # 图构建阈值：相关性超过0.6才建立连接
NORM_TYPE = 'standard'            # 数据归一化方式：'minmax'(0-1归一化) 或 'standard'(标准化)
TIME_ENCODING_ENABLED_IN_DATASET = True  # 是否启用时间编码：包含小时、星期等时间特征
TIME_FREQ_IN_DATASET = 'h'      # 时间频率：'h'(小时), 'd'(天), 'w'(周)

# --- TimeXer Model Parameters ---
# TimeXer模型架构参数
NEWS_PROCESSED_DIM = 64         # 新闻特征处理后的维度：将原始新闻特征压缩到64维
GCN_HIDDEN_DIM = 256           # GCN隐藏层维度：图卷积网络的中间层大小
GCN_OUTPUT_DIM = 128           # GCN输出维度：图卷积网络的输出特征维度
MLP_HIDDEN_DIM_1 = 256         # MLP第一隐藏层维度：减小以避免TimeXer特征主导
MLP_HIDDEN_DIM_2 = 256         # MLP第二隐藏层维度：多层感知机的第二层大小
NUM_CLASSES = 1 if TASK_TYPE == 'regression' else 2  # 输出类别数：回归任务=1，分类任务=2

# --- Training Parameters ---
# 训练参数配置
BATCH_SIZE = 32                    # 批次大小：每次训练使用32个样本
EPOCHS = 50                        # 训练轮数：最大训练50个epoch（可能因早停而提前结束）
LEARNING_RATE = 0.001             # 学习率：控制参数更新的步长，提高初始学习率
WEIGHT_DECAY = 1e-5               # 权重衰减：L2正则化系数，防止过拟合
VALIDATION_SPLIT_RATIO = 0.15     # 验证集比例：15%的数据用于验证
TEST_SPLIT_RATIO = 0.15           # 测试集比例：15%的数据用于最终测试
FORCE_RECOMPUTE_NEWS = False      # 是否强制重新计算新闻特征：False=使用缓存
RANDOM_SEED = 42                  # 随机种子：确保实验可重现

# --- Early Stopping Parameters ---
# 早停机制参数：防止过拟合，节省训练时间
EARLY_STOPPING_PATIENCE = 20     # 早停耐心值：连续20个epoch没有改善就停止训练
MIN_DELTA = 1e-6                  # 最小改善阈值：改善必须大于0.000001才算有效

def set_random_seeds(seed=42):
    """
    设置所有随机种子以确保结果可重现

    Args:
        seed (int): 随机种子值

    功能：
        1. 设置Python内置random模块的种子
        2. 设置NumPy随机数生成器的种子
        3. 设置PyTorch CPU随机数生成器的种子
        4. 设置PyTorch GPU随机数生成器的种子
        5. 确保CUDA操作的确定性
        6. 禁用CUDA的性能优化（为了确定性）
        7. 设置Python哈希随机化种子
    """
    random.seed(seed)                              # Python内置随机数
    np.random.seed(seed)                          # NumPy随机数
    torch.manual_seed(seed)                       # PyTorch CPU随机数
    torch.cuda.manual_seed(seed)                  # 当前GPU随机数
    torch.cuda.manual_seed_all(seed)              # 所有GPU随机数
    # 确保CUDA操作的确定性（可能会降低性能但保证可重现性）
    torch.backends.cudnn.deterministic = True     # 强制使用确定性算法
    torch.backends.cudnn.benchmark = False        # 禁用自动优化（为了确定性）
    # 设置Python哈希随机化环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)      # 确保字典等数据结构的顺序一致

# --- Dynamic File Path ---
model_variant = ['TimeXer', TASK_TYPE]
model_variant.append("with_gcn" if USE_GCN else "no_gcn")
model_variant.append("with_news" if USE_NEWS_FEATURES else "no_news")
model_variant_str = "_".join(model_variant)
BEST_MODEL_PATH = os.path.join(CACHE_DIR, f"{model_variant_str}_{BEST_MODEL_NAME}")
print(f"--- Configuration: {model_variant_str} ---")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

def save_test_predictions(all_preds, all_targets, coin_names, model_name):
    """
    保存测试集的预测值和真实值到CSV文件

    功能：
        将模型在测试集上的预测结果保存为两个CSV文件：
        1. 详细预测结果：每个样本每个币种的预测值、真实值和误差
        2. 统计信息：每个币种的统计指标汇总

    Args:
        all_preds: 所有预测值 [num_samples, num_coins]
        all_targets: 所有真实值 [num_samples, num_coins]
        coin_names: 币种名称列表
        model_name: 模型名称（用于文件命名）

    输出文件：
        - test_predictions_{model_name}.csv: 详细预测结果
        - test_statistics_{model_name}.csv: 统计信息汇总
    """
    # === 创建保存目录 ===
    save_dir = "experiments/caches/test_predictions"
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

    # === 保存详细预测结果 ===
    # 包含每个样本每个币种的详细预测信息
    predictions_file = os.path.join(save_dir, f"test_predictions_{model_name}.csv")
    with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['sample_idx', 'coin', 'true_value', 'predicted_value', 'absolute_error', 'percentage_error'])

        # === 遍历所有样本和币种 ===
        for sample_idx in range(len(all_preds)):
            for coin_idx, coin_name in enumerate(coin_names):
                # === 提取当前样本当前币种的预测值和真实值 ===
                true_val = all_targets[sample_idx, coin_idx]
                pred_val = all_preds[sample_idx, coin_idx]

                # === 计算误差指标 ===
                abs_error = abs(true_val - pred_val)  # 绝对误差
                # 计算百分比误差，避免除零错误
                pct_error = (abs_error / abs(true_val)) * 100 if abs(true_val) > 1e-8 else float('inf')

                # === 写入一行数据 ===
                writer.writerow([sample_idx, coin_name, true_val, pred_val, abs_error, pct_error])

    # === 保存统计信息 ===
    # 包含每个币种的统计指标汇总
    statistics_file = os.path.join(save_dir, f"test_statistics_{model_name}.csv")
    with open(statistics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头：包含均值、标准差、最值、MAE、MAPE等统计指标
        writer.writerow(['coin', 'mean_true', 'mean_pred', 'std_true', 'std_pred',
                        'min_true', 'min_pred', 'max_true', 'max_pred', 'mae', 'mape'])

        # === 遍历每个币种计算统计指标 ===
        for coin_idx, coin_name in enumerate(coin_names):
            # === 提取当前币种的所有预测值和真实值 ===
            true_vals = all_targets[:, coin_idx]
            pred_vals = all_preds[:, coin_idx]

            # === 计算评估指标 ===
            mae = mean_absolute_error(true_vals, pred_vals)  # 平均绝对误差
            # 计算平均绝对百分比误差，避免除零错误
            mape = np.mean(np.abs((true_vals - pred_vals) / np.where(np.abs(true_vals) > 1e-8, true_vals, 1e-8))) * 100

            # === 写入统计信息 ===
            writer.writerow([
                coin_name,                    # 币种名称
                np.mean(true_vals), np.mean(pred_vals),    # 均值
                np.std(true_vals), np.std(pred_vals),      # 标准差
                np.min(true_vals), np.min(pred_vals),      # 最小值
                np.max(true_vals), np.max(pred_vals),      # 最大值
                mae, mape                     # MAE和MAPE
            ])

    # === 打印保存信息 ===
    print(f"测试集预测结果已保存到:")
    print(f"  详细结果: {predictions_file}")
    print(f"  统计信息: {statistics_file}")

    return predictions_file, statistics_file

def evaluate_model(model, data_loader, criterion, edge_index, edge_weights, device, task_type, scaler=None):
    """
    模型评估函数

    功能：
        在给定数据集上评估模型性能，计算损失和各种评估指标

    Args:
        model: 要评估的模型
        data_loader: 数据加载器（验证集或测试集）
        criterion: 损失函数
        edge_index: 图的边索引
        edge_weights: 图的边权重
        device: 计算设备（CPU或GPU）
        task_type: 任务类型（'classification' 或 'regression'）
        scaler: 数据归一化器（用于反归一化）

    Returns:
        metrics: 包含各种评估指标的字典
        all_preds: 所有预测值
        all_targets: 所有真实值
    """
    # === 设置模型为评估模式 ===
    model.eval()  # 禁用dropout和batch normalization的训练行为
    total_loss, all_preds, all_targets = 0.0, [], []

    # === 禁用梯度计算以节省内存和加速 ===
    with torch.no_grad():
        # === 遍历数据加载器的每个批次 ===
        for batch_data in tqdm(data_loader, desc="Evaluating"):
            # === 提取批次数据并移动到指定设备 ===
            price_seq = batch_data['price_seq'].to(device)        # 价格序列数据
            target_data = batch_data['target_price'].to(device)   # 目标价格数据

            # === 处理可选的时间编码特征 ===
            x_mark_enc = batch_data.get('price_seq_mark')
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc.to(device)

            # === 处理可选的新闻特征 ===
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(device)

            # === 模型前向传播 ===
            outputs = model(
                price_seq,              # 价格序列
                x_mark_enc,            # 时间编码
                edge_index=edge_index,  # 图的边索引
                edge_weight=edge_weights,  # 图的边权重
                news_features=news_features  # 新闻特征
            )

            # === 根据任务类型处理目标和损失 ===
            if task_type == 'classification':
                # === 分类任务：将价格变化转换为涨跌标签 ===
                if PREDICTION_TARGET == 'price':
                    # 直接预测价格：判断价格是否大于某个阈值
                    targets = (target_data > 0).long()  # 涨=1, 跌=0
                elif PREDICTION_TARGET in ('diff', 'return'):
                    # 预测价差/变化率：数据集中 target 已是“下一步的差值/收益率”；直接判断正负
                    targets = (target_data > 0).long()  # 上涨=1, 下跌=0

                    # 调试信息：检查标签分布
                    if batch_idx == 0:
                        target_stats = {
                            'total_samples': target_data.numel(),
                            'positive_samples': (target_data > 0).sum().item(),
                            'negative_samples': (target_data <= 0).sum().item(),
                            'target_mean': target_data.mean().item(),
                            'target_std': target_data.std().item(),
                            'target_min': target_data.min().item(),
                            'target_max': target_data.max().item()
                        }
                        print(f"🔍 验证集标签统计: {target_stats}")
                        print(f"🔍 上涨比例: {target_stats['positive_samples']/target_stats['total_samples']:.1%}")
                        print(f"🔍 下跌比例: {target_stats['negative_samples']/target_stats['total_samples']:.1%}")
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
                    raise ValueError(f"Unexpected output shape in evaluate_model: {outputs.shape}")

                loss = criterion(outputs_flat, targets_flat)
                preds = torch.argmax(outputs, dim=-1)  # 获取预测类别

                # === 验证时的调试信息 ===
                if batch_idx == 0:  # 只在第一个batch打印验证调试信息
                    with torch.no_grad():
                        # 分析验证集的预测分布
                        pred_classes = preds
                        class_0_count = (pred_classes == 0).sum().item()
                        class_1_count = (pred_classes == 1).sum().item()
                        total_preds = pred_classes.numel()

                        print(f"🔍 验证集预测分布: 下跌={class_0_count}/{total_preds} ({class_0_count/total_preds:.1%}), 上涨={class_1_count}/{total_preds} ({class_1_count/total_preds:.1%})")
            else:
                # === 回归任务：仅支持价格预测 ===
                if PREDICTION_TARGET == 'price':
                    # 直接预测价格
                    targets = target_data
                else:
                    raise ValueError(f"PREDICTION_TARGET '{PREDICTION_TARGET}' not supported for regression")
                loss = criterion(outputs, targets)
                preds = outputs

            # === 累计损失和预测结果 ===
            total_loss += loss.item() * price_seq.size(0)  # 加权累计损失
            all_preds.append(preds.cpu().numpy())          # 收集预测值
            all_targets.append(targets.cpu().numpy())      # 收集真实值
            
    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = {'loss': avg_loss}

    # === 计算指标：根据任务类型计算不同的评估指标 ===
    if task_type == 'classification':
        # === 分类任务指标计算 ===
        # 对于分类任务，我们计算准确率、精确率、召回率、F1分数等指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

        # all_preds 是预测的类别 (0 或 1)
        # all_targets 是真实的类别 (0 或 1)

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

        all_targets = all_targets.squeeze(1)
        for i, coin_name in enumerate(COIN_NAMES):
            coin_targets = all_targets[:, i]
            coin_preds = all_preds[:, i]

            # 计算每个币种的分类指标
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
        
        # Initialize lists to store per-coin metrics
        coin_maes = []
        coin_mses = []
        coin_mapes = []
        coin_r2s = []
        
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
        
        # Calculate new MAE: sum of all true values / sum of all predicted values
        total_true_sum = np.sum(original_targets)
        total_pred_sum = np.sum(original_preds)
        new_mae = total_true_sum / total_pred_sum if total_pred_sum != 0 else float('inf')

        # Calculate overall metrics
        metrics.update({
            'mae': mean_absolute_error(original_targets, original_preds),
            'new_mae': new_mae,
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
    """
    主训练流程

    整体流程：
    1. 初始化设置（随机种子、设备、目录）
    2. 数据加载和预处理
    3. 图构建（如果使用GCN）
    4. 数据归一化
    5. 数据集创建和划分
    6. 模型初始化
    7. 训练循环（包含早停机制）
    8. 测试和结果保存
    """

    # === 步骤1: 初始化设置 ===
    # 设置随机种子确保结果可重现
    set_random_seeds(RANDOM_SEED)
    print(f"🎯 设置随机种子: {RANDOM_SEED}")
    print(f"📱 使用设备: {DEVICE}")
    # 创建缓存目录（如果不存在）
    os.makedirs(CACHE_DIR, exist_ok=True)

    # === 步骤2: 数据加载和预处理 ===
    print("📊 加载价格数据...")
    # 加载原始价格数据（CSV格式，时间为索引）
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    # 重命名列：从"BTC-USDT"格式改为"BTC"格式
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    # 选择需要的币种数据
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    # 确保时间索引是升序排列（从早到晚）
    if not price_df_full.index.is_monotonic_increasing:
        print(f"⚠️ 价格数据时间索引不是升序，正在排序...")
        price_df_full = price_df_full.sort_index()
        print(f"✅ 价格数据已按时间升序排列")

    # === 步骤3: 图构建（仅在使用GCN时） ===
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

    # === 步骤4: 数据预处理与归一化 ===
    print(f"🔄 数据预处理: target={PREDICTION_TARGET}, norm={NORM_TYPE}")

    # 先根据目标构造要归一化的数据
    if PREDICTION_TARGET == 'diff':
        df_to_scale = price_df_full.diff().dropna()
    elif PREDICTION_TARGET == 'return':
        # 使用收益率（百分比变化），更稳定
        df_to_scale = price_df_full.pct_change().dropna()
    else:  # 'price'
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
        values = scaler.fit_transform(df_to_scale)
        price_df = pd.DataFrame(values, columns=df_to_scale.columns, index=df_to_scale.index)
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

    # === 步骤5: 创建数据集和数据加载器 ===
    print("📰 加载新闻数据..." if USE_NEWS_FEATURES else "🚫 跳过新闻数据加载")

    # 根据配置决定是否加载新闻数据
    news_data_dict = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) if USE_NEWS_FEATURES else None

    if USE_NEWS_FEATURES:
        print(f"� 将从缓存文件加载预处理的新闻特征")
        print(f"📁 缓存路径: {os.path.join(CACHE_DIR, 'all_processed_news_feature_new10days.pt')}")

        # 检查缓存文件是否存在
        cache_file = os.path.join(CACHE_DIR, "all_processed_news_feature_new10days.pt")
        if os.path.exists(cache_file):
            print(f"✅ 新闻特征缓存文件存在")
        else:
            print(f"❌ 新闻特征缓存文件不存在: {cache_file}")
            print(f"� 请确保已预先生成新闻特征文件，或设置 USE_NEWS_FEATURES = False")

    if USE_NEWS_FEATURES:
        processed_news_path = os.path.join(CACHE_DIR, "all_processed_news_feature_new10days.pt")
        # 对于diff/return，先尝试自动对齐，失败时才重新计算
        if PREDICTION_TARGET in ('diff', 'return'):
            print(f"🔄 差分/变化率模式：将尝试自动对齐现有新闻特征")
            # 不强制重新计算，让数据集尝试自动对齐
            # FORCE_RECOMPUTE_NEWS = True

            # 临时方案：如果自动对齐失败，可以禁用新闻特征
            # USE_NEWS_FEATURES = False
            # print(f"⚠️  临时禁用新闻特征以避免索引不匹配问题")
    else:
        processed_news_path = None

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

    print(f"📊 数据集大小: {len(dataset)} 个样本")
    print(f"📊 训练集: {len(train_dataset)} 个样本")
    print(f"📊 验证集: {len(val_dataset)} 个样本")
    print(f"📊 测试集: {len(test_dataset)} 个样本")

    # # === 新闻特征分析 ===
    # if USE_NEWS_FEATURES:
    #     print(f"\n🔍 新闻特征详细分析:")
    #     print(f"  新闻特征维度: {dataset.news_feature_dim}")
    #     print(f"  新闻特征形状: {dataset.processed_news_features.shape}")

    #     # 获取一个样本查看新闻特征
    #     sample = dataset[10000]
    #     if 'news_features' in sample:
    #         news_sample = sample['news_features']
    #         print(f"  单个样本新闻特征形状: {news_sample.shape}")
    #         print(f"  新闻特征统计:")
    #         print(f"    均值: {news_sample.mean().item():.6f}")
    #         print(f"    标准差: {news_sample.std().item():.6f}")
    #         print(f"    最小值: {news_sample.min().item():.6f}")
    #         print(f"    最大值: {news_sample.max().item():.6f}")
    #         print(f"    零值比例: {(news_sample == 0).float().mean().item():.1%}")

    #         # 查看每个币种的新闻特征
    #         print(f"  各币种新闻特征强度:")
    #         for i, coin_name in enumerate(COIN_NAMES):
    #             coin_news = news_sample[i]
    #             coin_norm = torch.norm(coin_news).item()
    #             coin_nonzero = (coin_news != 0).sum().item()
    #             print(f"    {coin_name}: 范数={coin_norm:.4f}, 非零元素={coin_nonzero}/{len(coin_news)}")
    #     else:
    #         print(f"  ⚠️ 样本中没有新闻特征")
    # else:
    #     print(f"\n🚫 新闻特征已禁用")

    # 5. Initialize TimeXer Model
    print(f"🚀 初始化TimeXer模型，任务类型: {TASK_TYPE}, GCN配置: {GCN_CONFIG}")

    # 创建正确的配置对象
    class TimeXerConfigs:
        def __init__(self):
            self.enc_in = dataset.num_coins
            self.seq_len = PRICE_SEQ_LEN
            self.pred_len = 1
            self.d_model = 64
            self.d_ff = 128
            self.n_heads = 4
            self.e_layers = 2
            self.dropout = 0.3
            self.task_type = TASK_TYPE  # 重要：正确设置任务类型
            self.use_norm = True
            self.patch_len = 16
            self.stride = 8
            self.individual = False
            self.act = 'gelu'
            self.down_sampling_layers = 3
            self.down_sampling_window = 2
            self.down_sampling_method = 'avg'
            self.embed = 'timeF'
            self.freq = 'h'
            self.factor = 1
            self.output_attention = False
            self.activation = 'gelu'
            self.num_time_features = 6

    timexer_configs = TimeXerConfigs()

    model = UnifiedTimexerGCN(
        configs=timexer_configs,  # 传递正确的配置
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        use_gcn=USE_GCN,
        gcn_config=GCN_CONFIG,  # 新增：传递GCN配置
        news_feature_dim=dataset.news_feature_dim if USE_NEWS_FEATURES else None,
        news_processed_dim=NEWS_PROCESSED_DIM,
        mlp_hidden_dim_1=MLP_HIDDEN_DIM_1,
        mlp_hidden_dim_2=MLP_HIDDEN_DIM_2,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print(model)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # === 步骤6: 设置损失函数、优化器和学习率调度器 ===
    print("⚙️ 配置训练组件...")
    # 损失函数：根据任务类型选择
    if TASK_TYPE == 'classification':
        # # 添加类别权重来处理不平衡数据
        # class_weights = torch.tensor([0.4, 0.6]).to(DEVICE)  # [下跌权重, 上涨权重]
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        # print(f"🎯 使用加权交叉熵损失，类别权重: 下跌={class_weights[0]:.1f}, 上涨={class_weights[1]:.1f}")
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    # 优化器：Adam优化器，包含权重衰减（L2正则化）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 学习率调度器：当验证损失不再下降时自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,           # 要调整的优化器
        'min',              # 监控指标的模式：'min'表示越小越好
        patience=10,        # 等待10个epoch没有改善再降低学习率（增加耐心）
        factor=0.8,         # 学习率衰减因子：每次减少20%（更温和）
        min_lr=1e-7         # 最小学习率：防止学习率过小
    )
    print(f"📈 学习率调度器配置: patience=10, factor=0.8, min_lr=1e-7")

    # === 步骤7: 训练循环（包含早停机制） ===
    print(f"🚀 开始训练，最大轮数: {EPOCHS}")
    print(f"📊 早停配置: 耐心值={EARLY_STOPPING_PATIENCE}, 最小改善={MIN_DELTA}")

    # 早停机制变量
    if TASK_TYPE == 'classification':
        best_val_metric = 0.0  # F1分数越高越好
    else:
        best_val_metric = float('inf')  # 损失越低越好
    patience_counter = 0              # 耐心计数器（记录连续没有改善的epoch数）

    # 开始训练循环
    for epoch in range(EPOCHS):
        # === 7.1: 训练阶段 ===
        model.train()                 # 设置模型为训练模式（启用dropout、batch norm等）
        epoch_loss = 0.0             # 累计本epoch的训练损失

        # 创建进度条显示训练进度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")

        # === 遍历训练数据的每个批次 ===
        for batch_idx, batch_data in enumerate(train_pbar):
            # === 数据准备：从批次数据中提取输入特征，并移动到指定设备 ===
            price_seq = batch_data['price_seq'].to(DEVICE)        # 价格序列数据：[batch_size, seq_len, num_nodes]
            target_data = batch_data['target_price'].to(DEVICE)   # 目标价格数据：[batch_size, num_nodes]

            # === 处理可选的时间编码特征 ===
            # 时间编码包含小时、星期、月份等时间信息，帮助模型理解时间模式
            x_mark_enc = batch_data.get('price_seq_mark')
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc.to(DEVICE)

            # === 处理可选的新闻特征 ===
            # 新闻特征包含情感分析、关键词等信息，提供额外的市场信号
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(DEVICE)

            # === 前向传播阶段 ===
            optimizer.zero_grad()  # 清零上一步的梯度缓存，防止梯度累积

            # === 模型前向传播：多模态输入融合 ===
            # 将价格序列、时间编码、图结构、新闻特征输入模型
            outputs = model(
                price_seq,              # 价格序列：主要的时序特征
                x_mark_enc,            # 时间编码：时间模式特征
                edge_index=edge_index,  # 图的边索引：定义币种间连接关系
                edge_weight=edge_weights,  # 图的边权重：连接强度
                news_features=news_features  # 新闻特征：市场情感和事件信息
            )

            # === 损失计算：根据任务类型计算不同的损失 ===
            if TASK_TYPE == 'classification':
                # === 分类任务：预测价格涨跌方向 ===
                if PREDICTION_TARGET in ('diff', 'return'):
                    # 预测价差/变化率：数据集中 target 已经是“下一步的差值/收益率”；直接判断正负
                    targets = (target_data > 0).long()  # 上涨=1, 下跌=0

                    # 训练时的调试信息（每个epoch只打印一次）
                    if batch_idx == 0:
                        target_stats = {
                            'total_samples': target_data.numel(),
                            'positive_samples': (target_data > 0).sum().item(),
                            'negative_samples': (target_data <= 0).sum().item(),
                            'target_mean': target_data.mean().item(),
                            'target_std': target_data.std().item()
                        }
                        print(f"🔍 训练集标签统计: 上涨={target_stats['positive_samples']}/{target_stats['total_samples']} ({target_stats['positive_samples']/target_stats['total_samples']:.1%})")

                        # 检查模型输出分布
                        with torch.no_grad():
                            pred_probs = torch.softmax(outputs, dim=-1)
                            pred_classes = torch.argmax(outputs, dim=-1)
                            class_0_pred = (pred_classes == 0).sum().item()
                            class_1_pred = (pred_classes == 1).sum().item()
                            total_pred = pred_classes.numel()
                            print(f"🔍 模型预测分布: 下跌={class_0_pred}/{total_pred} ({class_0_pred/total_pred:.1%}), 上涨={class_1_pred}/{total_pred} ({class_1_pred/total_pred:.1%})")
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
                # 回归任务：仅支持价格预测
                if PREDICTION_TARGET == 'price':
                    # 直接预测价格
                    targets = target_data
                else:
                    raise ValueError(f"PREDICTION_TARGET '{PREDICTION_TARGET}' not supported for regression")
                loss = criterion(outputs, targets)

            # === 反向传播和参数更新 ===
            loss.backward()        # 计算梯度

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()       # 更新模型参数
            # 累计损失（加权平均，权重为批次大小）
            epoch_loss += loss.item() * price_seq.size(0)

        # === 7.2: 验证阶段 ===
        # 计算平均训练损失
        avg_train_loss = epoch_loss / len(train_dataset)
        # 在验证集上评估模型性能
        val_metrics, _, _ = evaluate_model(
            model, val_loader, criterion, edge_index, edge_weights, DEVICE, TASK_TYPE, scaler
        )
        # 选择不同的指标用于学习率调度和早停
        val_metric_for_scheduler = val_metrics['loss']  # 学习率调度仍使用损失
        if TASK_TYPE == 'classification':
            val_metric_for_early_stopping = val_metrics['f1_score']  # 早停使用F1分数
        else:
            val_metric_for_early_stopping = val_metrics['loss']  # 回归任务使用损失
        # 根据验证损失调整学习率
        scheduler.step(val_metric_for_scheduler)

        # === 7.3: 打印训练进度 ===
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📊 Epoch {epoch+1}/{EPOCHS} | 训练损失: {avg_train_loss:.4f} | 学习率: {current_lr:.6f}")

        # 检查学习率是否过小
        if current_lr < 1e-6:
            print(f"⚠️ 学习率过小 ({current_lr:.2e})，可能影响训练效果")

        print("--- 验证集指标 ---")
        for name, value in val_metrics.items():
            if not isinstance(value, dict):  # 跳过嵌套字典（如per_coin_metrics）
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

        # === 7.4: 早停机制和最佳模型保存 ===
        # 检查验证指标是否有显著改善
        if TASK_TYPE == 'classification':
            # 分类任务：F1分数越高越好
            if val_metric_for_early_stopping > best_val_metric + MIN_DELTA:
                best_val_metric = val_metric_for_early_stopping
                patience_counter = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"🚀 保存新的最佳模型到 {BEST_MODEL_PATH} (验证F1: {best_val_metric:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ 连续 {patience_counter} 个epoch无改善 (最佳F1: {best_val_metric:.4f})")
        else:
            # 回归任务：损失越低越好
            if val_metric_for_early_stopping < best_val_metric - MIN_DELTA:
                best_val_metric = val_metric_for_early_stopping
                patience_counter = 0
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"🚀 保存新的最佳模型到 {BEST_MODEL_PATH} (验证损失: {best_val_metric:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ 连续 {patience_counter} 个epoch无改善 (最佳损失: {best_val_metric:.4f})")
            print(f"⏳ 连续 {patience_counter} 个epoch无改善")

        # === 7.5: 早停检查 ===
        # 如果连续无改善的epoch数达到耐心值，触发早停
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 触发早停机制！训练在第 {epoch+1} 个epoch停止 (耐心值: {EARLY_STOPPING_PATIENCE})")
            print(f"💡 原因: 连续 {EARLY_STOPPING_PATIENCE} 个epoch验证指标无显著改善")
            break  # 跳出训练循环

    # === 步骤8: 测试阶段 ===
    print("\n" + "="*60)
    print("🧪 开始测试阶段 - 使用最佳模型")
    print("="*60)

    # 加载训练过程中保存的最佳模型
    if os.path.exists(BEST_MODEL_PATH):
        print(f"📂 加载最佳模型: {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("⚠️  警告: 未找到最佳模型文件，使用当前模型状态进行测试")

    # 在测试集上评估模型性能
    print("🔍 在测试集上评估模型...")
    test_metrics, test_preds, test_targets = evaluate_model(
        model, test_loader, criterion, edge_index, edge_weights, DEVICE, TASK_TYPE, scaler
    )

    # === 步骤9: 保存测试结果 ===
    print("💾 保存测试集预测结果...")
    if TASK_TYPE == 'regression':
        # 对于回归任务，需要将预测结果转换回原始尺度
        if PREDICTION_TARGET == 'price' and scaler:
            # 如果预测价格且使用了归一化，需要反归一化
            original_test_preds = scaler.inverse_transform(test_preds)      # 预测值反归一化
            original_test_targets = scaler.inverse_transform(test_targets)  # 真实值反归一化
        else:
            # 如果预测价格差或未使用归一化，直接使用原始值
            original_test_preds = test_preds
            original_test_targets = test_targets

        # 保存详细的预测结果到CSV文件
        save_test_predictions(original_test_preds, original_test_targets, COIN_NAMES, model_variant_str)

    # === 步骤10: 打印最终测试结果 ===
    print(f"\n" + "="*60)
    print("🎉 最终测试结果")
    print("="*60)
    print("📊 整体指标:")
    for name, value in test_metrics.items():
        if not isinstance(value, dict):  # 跳过嵌套字典
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
                elif name == 'avg_accuracy':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均准确率 - 各币种准确率的平均值")
                elif name == 'avg_precision':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均精确率 - 各币种精确率的平均值")
                elif name == 'avg_recall':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均召回率 - 各币种召回率的平均值")
                elif name == 'avg_f1_score':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均F1分数 - 各币种F1分数的平均值")
                elif name == 'precision_class_0':
                    print(f"    - {name.upper()}: {value:.4f}  # 下跌类精确率 - 预测下跌中实际下跌的比例")
                elif name == 'precision_class_1':
                    print(f"    - {name.upper()}: {value:.4f}  # 上涨类精确率 - 预测上涨中实际上涨的比例")
                elif name == 'recall_class_0':
                    print(f"    - {name.upper()}: {value:.4f}  # 下跌类召回率 - 实际下跌中被正确预测的比例")
                elif name == 'recall_class_1':
                    print(f"    - {name.upper()}: {value:.4f}  # 上涨类召回率 - 实际上涨中被正确预测的比例")
                elif name == 'f1_class_0':
                    print(f"    - {name.upper()}: {value:.4f}  # 下跌类F1分数 - 下跌类精确率和召回率的调和平均")
                elif name == 'f1_class_1':
                    print(f"    - {name.upper()}: {value:.4f}  # 上涨类F1分数 - 上涨类精确率和召回率的调和平均")
                elif name == 'loss':
                    print(f"    - {name.upper()}: {value:.4f}  # 测试损失 - 模型在测试集上的损失值")
                # 回归指标注释
                elif name == 'mae':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均绝对误差 - 预测值与真实值的平均绝对差")
                elif name == 'new_mae':
                    print(f"    - {name.upper()}: {value:.4f}  # 新MAE指标 - 真实值总和/预测值总和")
                elif name == 'mse':
                    print(f"    - {name.upper()}: {value:.4f}  # 均方误差 - 预测值与真实值差的平方的平均")
                elif name == 'rmse':
                    print(f"    - {name.upper()}: {value:.4f}  # 均方根误差 - MSE的平方根")
                elif name == 'r2':
                    print(f"    - {name.upper()}: {value:.4f}  # 决定系数 - 模型解释数据变异性的比例(越接近1越好)")
                elif name == 'mape':
                    print(f"    - {name.upper()}: {value:.4f}  # 平均绝对百分比误差 - 相对误差的百分比")
                elif name == 'normalized_mae':
                    print(f"    - {name.upper()}: {value:.4f}  # 归一化MAE - 消除币种价格尺度影响的MAE")
                elif name == 'normalized_mse':
                    print(f"    - {name.upper()}: {value:.4f}  # 归一化MSE - 消除币种价格尺度影响的MSE")
                elif name == 'normalized_rmse':
                    print(f"    - {name.upper()}: {value:.4f}  # 归一化RMSE - 消除币种价格尺度影响的RMSE")
                else:
                    print(f"    - {name.upper()}: {value:.4f}")
            elif isinstance(value, list):
                if name == 'confusion_matrix':
                    print(f"    - {name.upper()}: {value}  # 混淆矩阵")
                    # 安全地解析混淆矩阵
                    if len(value) >= 2 and len(value[0]) >= 2 and len(value[1]) >= 2:
                        print(f"      解释: 真负例={value[0][0]}, 假正例={value[0][1]}, 假负例={value[1][0]}, 真正例={value[1][1]}")
                    else:
                        print(f"      注意: 混淆矩阵维度不完整，可能某些类别未被预测到")
                else:
                    print(f"    - {name.upper()}: {value}")
            else:
                print(f"    - {name.upper()}: {value}")

    # 如果有每个币种的详细指标，也打印出来
    if 'per_coin_metrics' in test_metrics:
        print("\n📈 各币种详细指标:")
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
                    elif metric_name == 'mape':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的平均绝对百分比误差")
                    elif metric_name == 'normalized_mae':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的归一化MAE")
                    elif metric_name == 'normalized_mse':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的归一化MSE")
                    elif metric_name == 'normalized_rmse':
                        print(f"    - {metric_name.upper()}: {value:.4f}  # {coin_name}的归一化RMSE")
                    else:
                        print(f"    - {metric_name.upper()}: {value:.4f}")
                else:
                    print(f"    - {metric_name.upper()}: {value}")

    print(f"\n" + "="*60)
    print("✅ 训练脚本执行完成！")
    print("📁 检查 experiments/cache/ 目录查看保存的模型和结果")
    print("="*60)
