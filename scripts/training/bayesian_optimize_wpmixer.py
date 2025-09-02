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
import json
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# --- Model and Dataset Imports ---
from models.MixModel.unified_wpmixer import UnifiedWPMixer
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Master Switches ---
PREDICTION_TARGET = 'price'  # 只做价格回归
TASK_TYPE = 'regression'     # 固定为回归任务
USE_GCN = False             # 不使用GCN
USE_NEWS_FEATURES = False   # 不使用新闻特征

# --- Data & Cache Paths ---
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
CACHE_DIR = "experiments/cache/bayesian_optimization"
BEST_MODEL_NAME = "best_bayesian_wpmixer_model.pt"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
NORM_TYPE = 'standard'
TIME_ENCODING_ENABLED_IN_DATASET = True
TIME_FREQ_IN_DATASET = 'h'

# --- Fixed Training Parameters ---
RANDOM_SEED = 42

# --- Import Configuration ---
try:
    from bayesian_optimization_config import (
        OPTIMIZATION_OBJECTIVE, COMPOSITE_WEIGHTS, NORMALIZATION_PARAMS,
        N_CALLS, N_RANDOM_STARTS, EARLY_STOPPING_PATIENCE, MIN_DELTA,
        VALIDATION_SPLIT_RATIO, TEST_SPLIT_RATIO
    )
    print("✅ 已从配置文件加载优化设置")
except ImportError:
    print("⚠️ 未找到配置文件，使用默认设置")
    # --- Bayesian Optimization Configuration ---
    N_CALLS = 50  # 贝叶斯优化的迭代次数
    N_RANDOM_STARTS = 10  # 随机初始化的次数

    # --- Optimization Objective Configuration ---
    OPTIMIZATION_OBJECTIVE = 'composite'

    # 综合评分权重配置
    COMPOSITE_WEIGHTS = {
        'mse_weight': 0.4,      # MSE损失权重
        'mae_weight': 0.3,      # MAE权重
        'r2_weight': 0.2,       # R²权重（实际使用1-R²作为惩罚项）
        'mape_weight': 0.1      # MAPE权重
    }

    # 归一化参数
    NORMALIZATION_PARAMS = {
        'mse_scale': 100.0,     # MSE损失通常在0-100范围
        'mae_scale': 10.0,      # MAE通常在0-10范围
        'mape_scale': 100.0     # MAPE是百分比，通常在0-100范围
    }

    # --- Fixed Training Parameters ---
    VALIDATION_SPLIT_RATIO = 0.15
    TEST_SPLIT_RATIO = 0.15
    EARLY_STOPPING_PATIENCE = 15
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

def optimize_gpu_performance():
    """GPU性能优化设置"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        print("✅ GPU性能优化已启用")

# --- Bayesian Optimization Search Space ---
# 定义超参数搜索空间
search_space = [
    # WPMixer核心参数
    Integer(32, 256, name='d_model'),                    # 模型维度
    Integer(4, 16, name='patch_len'),                    # 补丁长度
    Integer(2, 8, name='patch_stride'),                  # 补丁步长
    Integer(30, 120, name='price_seq_len'),              # 价格序列长度
    Categorical(['db1', 'db4', 'db8', 'haar'], name='wavelet_name'),  # 小波类型
    Integer(1, 4, name='wavelet_level'),                 # 小波分解层数
    Integer(2, 8, name='tfactor'),                       # Token混合器扩展因子
    Integer(2, 8, name='dfactor'),                       # 嵌入混合器扩展因子
    
    # MLP参数
    Integer(256, 2048, name='mlp_hidden_dim_1'),         # MLP第一隐藏层维度
    Integer(128, 1024, name='mlp_hidden_dim_2'),         # MLP第二隐藏层维度
    
    # 训练参数
    Integer(16, 128, name='batch_size'),                 # 批次大小
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),  # 学习率
    Real(1e-6, 1e-2, prior='log-uniform', name='weight_decay'),   # 权重衰减
    Real(0.0, 0.5, name='dropout'),                      # Dropout率
    Integer(20, 100, name='epochs'),                     # 训练轮数
]

# 提取参数名称用于后续使用
param_names = [dim.name for dim in search_space]

def evaluate_model_performance(model, data_loader, criterion, device, scaler=None):
    """评估模型性能"""
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    
    with torch.no_grad():
        for batch_data in data_loader:
            price_seq = batch_data['price_seq'].to(device)
            target_data = batch_data['target_price'].to(device)
            
            outputs = model(price_data=price_seq)
            outputs = outputs.squeeze(-1)  # [batch, num_coins, 1] -> [batch, num_coins]
            
            loss = criterion(outputs, target_data)
            total_loss += loss.item() * price_seq.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(target_data.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 反归一化用于计算真实指标
    if scaler:
        original_preds = scaler.inverse_transform(all_preds)
        original_targets = scaler.inverse_transform(all_targets)
    else:
        original_preds = all_preds
        original_targets = all_targets
    
    # 计算评估指标
    mae = mean_absolute_error(original_targets, original_preds)
    mse = mean_squared_error(original_targets, original_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(original_targets, original_preds)
    mape = mean_absolute_percentage_error(original_targets, original_preds)
    
    metrics = {
        'loss': avg_loss,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    return metrics, all_preds, all_targets

def calculate_optimization_score(metrics, objective_type='composite'):
    """
    计算优化评分

    Args:
        metrics: 包含各种评估指标的字典
        objective_type: 优化目标类型

    Returns:
        score: 优化评分（越小越好）
        details: 评分详情字典
    """
    val_loss = metrics['loss']      # MSE损失
    val_mae = metrics['mae']        # 平均绝对误差
    val_r2 = metrics['r2']          # 决定系数
    val_mape = metrics['mape']      # 平均绝对百分比误差

    if objective_type == 'mse_only':
        # 仅优化MSE损失
        score = val_loss
        details = {
            'score_type': 'MSE Loss Only',
            'mse_loss': val_loss,
            'final_score': score
        }

    elif objective_type == 'mae_focused':
        # 主要优化MAE，辅助考虑R²
        r2_penalty = max(0, 1 - val_r2)
        score = 0.8 * val_mae + 0.2 * r2_penalty
        details = {
            'score_type': 'MAE Focused',
            'mae': val_mae,
            'r2_penalty': r2_penalty,
            'final_score': score
        }

    elif objective_type == 'r2_focused':
        # 主要优化R²，辅助考虑MSE
        r2_penalty = max(0, 1 - val_r2)
        normalized_mse = val_loss / NORMALIZATION_PARAMS['mse_scale']
        score = 0.7 * r2_penalty + 0.3 * normalized_mse
        details = {
            'score_type': 'R² Focused',
            'r2_penalty': r2_penalty,
            'normalized_mse': normalized_mse,
            'final_score': score
        }

    else:  # 'composite'
        # 综合优化多个指标
        normalized_loss = val_loss / NORMALIZATION_PARAMS['mse_scale']
        normalized_mae = val_mae / NORMALIZATION_PARAMS['mae_scale']
        normalized_r2_penalty = max(0, 1 - val_r2)
        normalized_mape = min(val_mape / NORMALIZATION_PARAMS['mape_scale'], 1.0)

        score = (
            COMPOSITE_WEIGHTS['mse_weight'] * normalized_loss +
            COMPOSITE_WEIGHTS['mae_weight'] * normalized_mae +
            COMPOSITE_WEIGHTS['r2_weight'] * normalized_r2_penalty +
            COMPOSITE_WEIGHTS['mape_weight'] * normalized_mape
        )

        details = {
            'score_type': 'Composite Score',
            'normalized_mse': normalized_loss,
            'normalized_mae': normalized_mae,
            'r2_penalty': normalized_r2_penalty,
            'normalized_mape': normalized_mape,
            'weights': COMPOSITE_WEIGHTS,
            'final_score': score
        }

    return score, details

class WPMixerConfigs:
    """WPMixer配置类"""
    def __init__(self, **kwargs):
        # 从kwargs中设置属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # 设置固定属性
        self.pred_length = 1
        self.no_decomposition = False
        self.use_amp = False
        self.task_type = TASK_TYPE
        self.device = DEVICE

# 全局变量用于存储数据集
global_dataset = None
global_train_loader = None
global_val_loader = None
global_test_loader = None
global_scaler = None

def prepare_data():
    """准备数据集（只执行一次）"""
    global global_dataset, global_train_loader, global_val_loader, global_test_loader, global_scaler

    if global_dataset is not None:
        return  # 数据已经准备好了

    print("📊 准备数据集...")

    # 加载价格数据
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    # 确保时间索引是升序排列
    if not price_df_full.index.is_monotonic_increasing:
        price_df_full = price_df_full.sort_index()

    # 创建数据集（使用默认序列长度）
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df_full,
        news_data_dict=None,
        seq_len=60,  # 默认值，会在优化过程中动态调整
        processed_news_features_path=None,
        force_recompute_news=False,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET,
        time_freq=TIME_FREQ_IN_DATASET,
    )

    # 数据集划分
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    # 数据标准化
    if NORM_TYPE != 'none':
        if NORM_TYPE == 'standard':
            scaler = StandardScaler()
        elif NORM_TYPE == 'minmax':
            scaler = MinMaxScaler()

        # 只使用训练集数据来拟合标准化器
        train_data_for_scaling = []
        for idx in train_indices:
            sample = dataset[idx]
            price_seq = sample['price_seq'].numpy()
            target_price = sample['target_price'].numpy()
            train_data_for_scaling.append(price_seq)
            train_data_for_scaling.append(target_price.reshape(1, -1))

        train_data_array = np.vstack(train_data_for_scaling)
        scaler.fit(train_data_array)

        # 更新数据集
        original_price_df = dataset.price_data_df.copy()
        scaled_values = scaler.transform(original_price_df.values)
        dataset.price_data_df = pd.DataFrame(
            scaled_values,
            columns=original_price_df.columns,
            index=original_price_df.index
        )
        global_scaler = scaler
    else:
        global_scaler = None

    # 创建数据子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 存储全局变量 - 修复：正确设置全局变量
    global_dataset = dataset
    global_train_loader = train_dataset
    global_val_loader = val_dataset
    global_test_loader = test_dataset

    print(f"✅ 数据准备完成: 训练集={len(train_dataset)}, 验证集={len(val_dataset)}, 测试集={len(test_dataset)}")

    # 返回数据集以便外部访问
    return dataset, train_dataset, val_dataset, test_dataset, scaler

def objective(params_list):
    """贝叶斯优化的目标函数"""
    global global_dataset, global_train_loader, global_val_loader, global_scaler

    # 将参数列表转换为字典，并确保类型正确
    params = {}
    for i, name in enumerate(param_names):
        value = params_list[i]
        # 转换numpy类型为Python原生类型
        if hasattr(value, 'item'):
            value = value.item()
        params[name] = value

    try:
        # 设置随机种子
        set_random_seeds(RANDOM_SEED)

        print(f"\n🔍 评估参数组合: {params}")

        # 确保数据已准备
        if global_dataset is None:
            prepare_data()

        # 更新数据集的序列长度
        if global_dataset.seq_len != params['price_seq_len']:
            global_dataset.seq_len = params['price_seq_len']
            print(f"📏 更新序列长度为: {params['price_seq_len']}")

        # 创建数据加载器，确保batch_size是整数
        batch_size = int(params['batch_size'])
        train_loader = DataLoader(global_train_loader, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(global_val_loader, batch_size=batch_size, shuffle=False)

        # 创建模型配置
        configs = WPMixerConfigs(
            input_length=params['price_seq_len'],
            num_coins=global_dataset.num_coins,
            d_model=params['d_model'],
            patch_len=params['patch_len'],
            patch_stride=params['patch_stride'],
            wavelet_name=params['wavelet_name'],
            level=params['wavelet_level'],
            tfactor=params['tfactor'],
            dfactor=params['dfactor'],
            dropout=params['dropout']
        )

        # 创建模型
        model = UnifiedWPMixer(
            configs=configs,
            use_gcn=USE_GCN,
            gcn_config='improved_light',
            news_feature_dim=None,
            gcn_hidden_dim=256,
            gcn_output_dim=128,
            news_processed_dim=64,
            mlp_hidden_dim_1=params['mlp_hidden_dim_1'],
            mlp_hidden_dim_2=params['mlp_hidden_dim_2'],
            num_classes=1
        ).to(DEVICE)

        # 设置训练组件
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.8, min_lr=1e-7)

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(params['epochs']):
            # 训练阶段
            model.train()
            epoch_loss = 0.0

            for batch_data in train_loader:
                price_seq = batch_data['price_seq'].to(DEVICE)
                target_data = batch_data['target_price'].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(price_data=price_seq)
                outputs = outputs.squeeze(-1)

                loss = criterion(outputs, target_data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item() * price_seq.size(0)

            # 验证阶段
            val_metrics, _, _ = evaluate_model_performance(model, val_loader, criterion, DEVICE, global_scaler)
            val_loss = val_metrics['loss']

            scheduler.step(val_loss)

            # 早停检查
            if val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"⏹️ 早停在第 {epoch+1} 轮")
                break

        # 获取最终验证指标进行综合评估
        final_val_metrics, _, _ = evaluate_model_performance(model, val_loader, criterion, DEVICE, global_scaler)

        # 计算优化评分
        optimization_score, score_details = calculate_optimization_score(final_val_metrics, OPTIMIZATION_OBJECTIVE)

        # 打印详细的评估结果
        print(f"📊 最终验证指标:")
        print(f"   MSE Loss: {final_val_metrics['loss']:.6f}")
        print(f"   MAE: {final_val_metrics['mae']:.6f}")
        print(f"   R²: {final_val_metrics['r2']:.6f}")
        print(f"   MAPE: {final_val_metrics['mape']:.6f}")

        print(f"📊 优化评分详情 ({score_details['score_type']}):")
        if OPTIMIZATION_OBJECTIVE == 'composite':
            print(f"   归一化MSE: {score_details['normalized_mse']:.6f} (权重{COMPOSITE_WEIGHTS['mse_weight']:.1%})")
            print(f"   归一化MAE: {score_details['normalized_mae']:.6f} (权重{COMPOSITE_WEIGHTS['mae_weight']:.1%})")
            print(f"   R²惩罚项: {score_details['r2_penalty']:.6f} (权重{COMPOSITE_WEIGHTS['r2_weight']:.1%})")
            print(f"   归一化MAPE: {score_details['normalized_mape']:.6f} (权重{COMPOSITE_WEIGHTS['mape_weight']:.1%})")
        elif OPTIMIZATION_OBJECTIVE == 'mae_focused':
            print(f"   MAE: {score_details['mae']:.6f} (权重80%)")
            print(f"   R²惩罚项: {score_details['r2_penalty']:.6f} (权重20%)")
        elif OPTIMIZATION_OBJECTIVE == 'r2_focused':
            print(f"   R²惩罚项: {score_details['r2_penalty']:.6f} (权重70%)")
            print(f"   归一化MSE: {score_details['normalized_mse']:.6f} (权重30%)")
        else:  # mse_only
            print(f"   MSE损失: {score_details['mse_loss']:.6f}")

        print(f"   最终优化评分: {optimization_score:.6f}")

        # 确保返回值是有限的
        if np.isnan(optimization_score) or np.isinf(optimization_score):
            print(f"⚠️ 检测到无效优化评分，返回大数值")
            return 1e6

        return float(optimization_score)

    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1e6  # 返回大数值而不是无穷大

def save_optimization_results(result, best_params, best_score, test_metrics=None):
    """保存优化结果"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 保存详细结果
    results_file = os.path.join(CACHE_DIR, f"bayesian_optimization_results_{timestamp}.json")

    results_data = {
        'best_params': best_params,
        'best_score': best_score,
        'test_metrics': test_metrics,
        'optimization_objective': OPTIMIZATION_OBJECTIVE,
        'composite_weights': COMPOSITE_WEIGHTS if OPTIMIZATION_OBJECTIVE == 'composite' else None,
        'optimization_history': {
            'func_vals': result.func_vals.tolist(),
            'x_iters': [dict(zip(param_names, x)) for x in result.x_iters],
        },
        'search_space': {dim.name: str(dim) for dim in search_space},
        'n_calls': N_CALLS,
        'n_random_starts': N_RANDOM_STARTS,
        'timestamp': datetime.now().isoformat()
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"💾 优化结果已保存到: {results_file}")

    # 保存最佳参数的简化版本
    best_params_file = os.path.join(CACHE_DIR, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)

    # 保存最佳参数的Python配置文件格式
    best_params_py_file = os.path.join(CACHE_DIR, f"best_params_{timestamp}.py")
    save_best_params_as_python_config(best_params, best_score, test_metrics, best_params_py_file)

    # 保存最佳参数的YAML格式（便于阅读）
    best_params_yaml_file = os.path.join(CACHE_DIR, f"best_params_{timestamp}.yaml")
    save_best_params_as_yaml(best_params, best_score, test_metrics, best_params_yaml_file)

    return results_file, best_params_file, best_params_py_file, best_params_yaml_file

def save_best_params_as_python_config(best_params, best_score, test_metrics, filepath):
    """将最佳参数保存为Python配置文件格式"""

    config_content = f'''"""
最佳贝叶斯优化参数配置
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
优化目标: {OPTIMIZATION_OBJECTIVE}
最佳评分: {best_score:.6f}
"""

# =============================================================================
# 最佳超参数配置
# =============================================================================

# WPMixer核心参数
D_MODEL = {best_params['d_model']}
PATCH_LEN = {best_params['patch_len']}
PATCH_STRIDE = {best_params['patch_stride']}
PRICE_SEQ_LEN = {best_params['price_seq_len']}
WAVELET_NAME = '{best_params['wavelet_name']}'
WAVELET_LEVEL = {best_params['wavelet_level']}
TFACTOR = {best_params['tfactor']}
DFACTOR = {best_params['dfactor']}

# MLP架构参数
MLP_HIDDEN_DIM_1 = {best_params['mlp_hidden_dim_1']}
MLP_HIDDEN_DIM_2 = {best_params['mlp_hidden_dim_2']}

# 训练参数
BATCH_SIZE = {best_params['batch_size']}
LEARNING_RATE = {best_params['learning_rate']:.8f}
WEIGHT_DECAY = {best_params['weight_decay']:.8f}
DROPOUT = {best_params['dropout']:.6f}
EPOCHS = {best_params['epochs']}

# =============================================================================
# 优化结果
# =============================================================================

OPTIMIZATION_SCORE = {best_score:.6f}
OPTIMIZATION_OBJECTIVE = '{OPTIMIZATION_OBJECTIVE}'
'''

    if OPTIMIZATION_OBJECTIVE == 'composite':
        config_content += f'''
COMPOSITE_WEIGHTS = {{
    'mse_weight': {COMPOSITE_WEIGHTS['mse_weight']},
    'mae_weight': {COMPOSITE_WEIGHTS['mae_weight']},
    'r2_weight': {COMPOSITE_WEIGHTS['r2_weight']},
    'mape_weight': {COMPOSITE_WEIGHTS['mape_weight']}
}}
'''

    if test_metrics:
        config_content += f'''
# 测试集性能指标
TEST_METRICS = {{
    'loss': {test_metrics.get('loss', 0):.6f},
    'mae': {test_metrics.get('mae', 0):.6f},
    'mse': {test_metrics.get('mse', 0):.6f},
    'rmse': {test_metrics.get('rmse', 0):.6f},
    'r2': {test_metrics.get('r2', 0):.6f},
    'mape': {test_metrics.get('mape', 0):.6f}
}}
'''

    config_content += '''
# =============================================================================
# 使用示例
# =============================================================================

def get_wpmixer_config():
    """获取WPMixer配置对象"""
    class WPMixerConfigs:
        def __init__(self):
            self.input_length = PRICE_SEQ_LEN
            self.pred_length = 1
            self.num_coins = 8
            self.d_model = D_MODEL
            self.patch_len = PATCH_LEN
            self.patch_stride = PATCH_STRIDE
            self.wavelet_name = WAVELET_NAME
            self.level = WAVELET_LEVEL
            self.tfactor = TFACTOR
            self.dfactor = DFACTOR
            self.no_decomposition = False
            self.use_amp = False
            self.dropout = DROPOUT
            self.task_type = 'regression'
            self.device = 'cuda'

    return WPMixerConfigs()

def get_training_config():
    """获取训练配置"""
    return {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'epochs': EPOCHS,
        'dropout': DROPOUT
    }

if __name__ == '__main__':
    print("🎯 最佳贝叶斯优化参数")
    print(f"优化评分: {OPTIMIZATION_SCORE:.6f}")
    print(f"优化目标: {OPTIMIZATION_OBJECTIVE}")

    config = get_wpmixer_config()
    training_config = get_training_config()

    print("\\n📋 WPMixer配置:")
    for attr in dir(config):
        if not attr.startswith('_'):
            print(f"  {attr}: {getattr(config, attr)}")

    print("\\n🏃 训练配置:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
'''

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(config_content)

    print(f"📄 Python配置文件已保存到: {filepath}")

def save_best_params_as_yaml(best_params, best_score, test_metrics, filepath):
    """将最佳参数保存为YAML格式"""

    yaml_content = f'''# 最佳贝叶斯优化参数配置
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 优化目标: {OPTIMIZATION_OBJECTIVE}
# 最佳评分: {best_score:.6f}

optimization_info:
  score: {best_score:.6f}
  objective: "{OPTIMIZATION_OBJECTIVE}"
  timestamp: "{datetime.now().isoformat()}"
'''

    if OPTIMIZATION_OBJECTIVE == 'composite':
        yaml_content += f'''  composite_weights:
    mse_weight: {COMPOSITE_WEIGHTS['mse_weight']}
    mae_weight: {COMPOSITE_WEIGHTS['mae_weight']}
    r2_weight: {COMPOSITE_WEIGHTS['r2_weight']}
    mape_weight: {COMPOSITE_WEIGHTS['mape_weight']}
'''

    yaml_content += f'''
# WPMixer核心参数
wpmixer:
  d_model: {best_params['d_model']}
  patch_len: {best_params['patch_len']}
  patch_stride: {best_params['patch_stride']}
  price_seq_len: {best_params['price_seq_len']}
  wavelet_name: "{best_params['wavelet_name']}"
  wavelet_level: {best_params['wavelet_level']}
  tfactor: {best_params['tfactor']}
  dfactor: {best_params['dfactor']}

# MLP架构参数
mlp:
  hidden_dim_1: {best_params['mlp_hidden_dim_1']}
  hidden_dim_2: {best_params['mlp_hidden_dim_2']}

# 训练参数
training:
  batch_size: {best_params['batch_size']}
  learning_rate: {best_params['learning_rate']:.8f}
  weight_decay: {best_params['weight_decay']:.8f}
  dropout: {best_params['dropout']:.6f}
  epochs: {best_params['epochs']}
'''

    if test_metrics:
        yaml_content += f'''
# 测试集性能指标
test_metrics:
  loss: {test_metrics.get('loss', 0):.6f}
  mae: {test_metrics.get('mae', 0):.6f}
  mse: {test_metrics.get('mse', 0):.6f}
  rmse: {test_metrics.get('rmse', 0):.6f}
  r2: {test_metrics.get('r2', 0):.6f}
  mape: {test_metrics.get('mape', 0):.6f}
'''

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"📄 YAML配置文件已保存到: {filepath}")

def train_final_model(best_params):
    """使用最佳参数训练最终模型"""
    global global_dataset, global_train_loader, global_val_loader, global_test_loader, global_scaler

    print(f"\n🚀 使用最佳参数训练最终模型...")
    print(f"📋 最佳参数: {best_params}")

    # 设置随机种子
    set_random_seeds(RANDOM_SEED)

    # 更新数据集序列长度
    global_dataset.seq_len = best_params['price_seq_len']

    # 创建数据加载器
    train_loader = DataLoader(global_train_loader, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(global_val_loader, batch_size=best_params['batch_size'], shuffle=False)
    test_loader = DataLoader(global_test_loader, batch_size=best_params['batch_size'], shuffle=False)

    # 创建最终模型
    configs = WPMixerConfigs(
        input_length=best_params['price_seq_len'],
        num_coins=global_dataset.num_coins,
        d_model=best_params['d_model'],
        patch_len=best_params['patch_len'],
        patch_stride=best_params['patch_stride'],
        wavelet_name=best_params['wavelet_name'],
        level=best_params['wavelet_level'],
        tfactor=best_params['tfactor'],
        dfactor=best_params['dfactor'],
        dropout=best_params['dropout']
    )

    model = UnifiedWPMixer(
        configs=configs,
        use_gcn=USE_GCN,
        gcn_config='improved_light',
        news_feature_dim=None,
        gcn_hidden_dim=256,
        gcn_output_dim=128,
        news_processed_dim=64,
        mlp_hidden_dim_1=best_params['mlp_hidden_dim_1'],
        mlp_hidden_dim_2=best_params['mlp_hidden_dim_2'],
        num_classes=1
    ).to(DEVICE)

    # 训练设置
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8, min_lr=1e-7)

    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(CACHE_DIR, BEST_MODEL_NAME)

    print(f"🏃 开始训练最终模型，最大轮数: {best_params['epochs']}")

    for epoch in range(best_params['epochs']):
        # 训练阶段
        model.train()
        epoch_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{best_params['epochs']}")
        for batch_data in train_pbar:
            price_seq = batch_data['price_seq'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(price_data=price_seq)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, target_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * price_seq.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        # 验证阶段
        val_metrics, _, _ = evaluate_model_performance(model, val_loader, criterion, DEVICE, global_scaler)
        val_loss = val_metrics['loss']

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: 训练损失={epoch_loss/len(global_train_loader):.6f}, 验证损失={val_loss:.6f}, R2={val_metrics['r2']:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 保存最佳模型")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"⏹️ 早停在第 {epoch+1} 轮")
            break

    # 加载最佳模型并测试
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))

    # 测试阶段
    print(f"\n🧪 测试最终模型...")
    test_metrics, _, _ = evaluate_model_performance(model, test_loader, criterion, DEVICE, global_scaler)

    print(f"\n🎉 最终测试结果:")
    for name, value in test_metrics.items():
        print(f"  {name.upper()}: {value:.6f}")

    return model, test_metrics

if __name__ == '__main__':
    """
    贝叶斯优化WPMixer主流程

    流程：
    1. 初始化设置和数据准备
    2. 运行贝叶斯优化寻找最佳超参数
    3. 使用最佳参数训练最终模型
    4. 评估和保存结果
    """

    print("🎯 开始贝叶斯优化WPMixer")
    print("="*60)

    # 初始化设置
    set_random_seeds(RANDOM_SEED)
    print(f"🎲 设置随机种子: {RANDOM_SEED}")
    print(f"📱 使用设备: {DEVICE}")

    # GPU性能优化
    optimize_gpu_performance()

    # 创建缓存目录
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"📁 缓存目录: {CACHE_DIR}")

    # 准备数据
    prepare_data()

    # 打印搜索空间信息
    print(f"\n🔍 贝叶斯优化配置:")
    print(f"  迭代次数: {N_CALLS}")
    print(f"  随机初始化: {N_RANDOM_STARTS}")
    print(f"  搜索空间维度: {len(search_space)}")
    print(f"  优化目标: {OPTIMIZATION_OBJECTIVE}")

    if OPTIMIZATION_OBJECTIVE == 'composite':
        print(f"  综合评分权重:")
        print(f"    MSE损失: {COMPOSITE_WEIGHTS['mse_weight']:.1%}")
        print(f"    MAE: {COMPOSITE_WEIGHTS['mae_weight']:.1%}")
        print(f"    R²: {COMPOSITE_WEIGHTS['r2_weight']:.1%}")
        print(f"    MAPE: {COMPOSITE_WEIGHTS['mape_weight']:.1%}")

    print(f"\n📊 搜索空间:")
    for dim in search_space:
        print(f"  {dim.name}: {dim}")

    # 运行贝叶斯优化
    print(f"\n🚀 开始贝叶斯优化...")
    start_time = datetime.now()

    try:
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=N_CALLS,
            n_random_starts=N_RANDOM_STARTS,
            acq_func='EI',  # Expected Improvement
            random_state=RANDOM_SEED,
            verbose=True
        )

        end_time = datetime.now()
        optimization_time = end_time - start_time

        print(f"\n✅ 贝叶斯优化完成!")
        print(f"⏱️ 优化耗时: {optimization_time}")
        print(f"🎯 最佳验证损失: {result.fun:.6f}")

        # 提取最佳参数
        best_params = dict(zip(param_names, result.x))

        # 确保所有参数都是Python原生类型，以避免后续出现类型错误
        for name, value in best_params.items():
            if hasattr(value, 'item'):
                best_params[name] = value.item()
        print(f"\n🏆 最佳参数组合:")
        for name, value in best_params.items():
            print(f"  {name}: {value}")

        # 使用最佳参数训练最终模型
        final_model, test_metrics = train_final_model(best_params)

        # 保存优化结果（包含测试指标）
        results_file, best_params_file, best_params_py_file, best_params_yaml_file = save_optimization_results(
            result, best_params, result.fun, test_metrics
        )

        # 保存最终结果摘要
        summary_file = os.path.join(CACHE_DIR, "optimization_summary.json")
        summary_data = {
            'optimization_completed': True,
            'best_validation_loss': result.fun,
            'best_params': best_params,
            'test_metrics': test_metrics,
            'optimization_time_seconds': optimization_time.total_seconds(),
            'n_calls': N_CALLS,
            'timestamp': datetime.now().isoformat()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\n📋 优化摘要已保存到: {summary_file}")
        print(f"📊 详细结果已保存到: {results_file}")
        print(f"🎯 最佳参数文件:")
        print(f"  JSON格式: {best_params_file}")
        print(f"  Python配置: {best_params_py_file}")
        print(f"  YAML格式: {best_params_yaml_file}")

        # 打印优化历史的简要统计
        print(f"\n📈 优化历史统计:")
        print(f"  最佳损失: {min(result.func_vals):.6f}")
        print(f"  最差损失: {max(result.func_vals):.6f}")
        print(f"  平均损失: {np.mean(result.func_vals):.6f}")
        print(f"  损失标准差: {np.std(result.func_vals):.6f}")

        # 找出前5个最佳配置
        sorted_indices = np.argsort(result.func_vals)
        print(f"\n🏅 前5个最佳配置:")
        for i, idx in enumerate(sorted_indices[:5]):
            params_dict = dict(zip(param_names, result.x_iters[idx]))
            print(f"  #{i+1}: 损失={result.func_vals[idx]:.6f}")
            print(f"       参数: {params_dict}")

        print(f"\n🎉 贝叶斯优化完成! 最佳模型已保存到: {os.path.join(CACHE_DIR, BEST_MODEL_NAME)}")

    except Exception as e:
        print(f"❌ 贝叶斯优化失败: {e}")
        import traceback
        traceback.print_exc()

        # 保存失败信息
        error_file = os.path.join(CACHE_DIR, f"optimization_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(error_file, 'w') as f:
            f.write(f"Optimization failed at: {datetime.now()}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Traceback:\n{traceback.format_exc()}")

        print(f"💾 错误信息已保存到: {error_file}")
        raise
