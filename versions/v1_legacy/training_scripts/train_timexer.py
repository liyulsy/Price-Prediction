import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import csv
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# Import the unified Timexer model and dataset components
from models.MixModel.unified_timexer_gcn import UnifiedTimexerGCN
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index

# --- Configuration and Hyperparameters ---
# Master switches to control the model architecture and task
TASK_TYPE = 'regression' # 'classification' or 'regression'
PREDICTION_TARGET = 'price' # 'price' or 'diff'
USE_NEWS_FEATURES = True
USE_GCN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Paths ---
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
NEWS_FEATURES_FOLDER = 'scripts/analysis/crypto_new_analyzer/features'
CACHE_DIR = "experiments/cache"
BEST_MODEL_NAME = "best_timexer_model.pt"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 90  # 贝叶斯优化得到的最佳序列长度
THRESHOLD = 0.6
NORM_TYPE = 'minmax'  # 改为 MinMaxScaler，更适合回归任务
TIME_ENCODING_ENABLED_IN_DATASET = True
TIME_FREQ_IN_DATASET = 'h'

# --- Model Parameters ---
NEWS_PROCESSED_DIM = 64  # 增加维度
GCN_HIDDEN_DIM = 256  # 增加 GCN 容量
GCN_OUTPUT_DIM = 128
MLP_HIDDEN_DIM_1 = 512  # 贝叶斯优化得到的最佳参数
MLP_HIDDEN_DIM_2 = 128  # 贝叶斯优化得到的最佳参数
NUM_CLASSES = 2

# TimeXerConfigs parameters (贝叶斯优化得到的最佳参数)
D_MODEL = 256  # 贝叶斯优化得到的最佳参数
PATCH_LEN = 24  # 贝叶斯优化得到的最佳参数
STRIDE = 4  # 贝叶斯优化得到的最佳参数
E_LAYERS = 3  # 贝叶斯优化得到的最佳参数
N_HEADS = 4  # 贝叶斯优化得到的最佳参数
D_FF = 512  # 贝叶斯优化得到的最佳参数
DROPOUT_TIMEXER = 0.15  # 贝叶斯优化得到的最佳参数

# --- Training Parameters ---
BATCH_SIZE = 32  # 增加批次大小，提高训练稳定性
EPOCHS = 20  # 增加训练轮数
LEARNING_RATE = 0.00015435211146627072  # 贝叶斯优化得到的最佳学习率
WEIGHT_DECAY = 3.0864039085286224e-05  # 贝叶斯优化得到的最佳权重衰减
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# --- Dynamic parameter based on the master switch ---
model_variant = ['TIMEXER', TASK_TYPE]
if USE_NEWS_FEATURES: model_variant.append("with_news")
else: model_variant.append("no_news")
if USE_GCN: model_variant.append("with_gcn")
else: model_variant.append("no_gcn")

model_variant_str = "_".join(model_variant)
BEST_MODEL_PATH = os.path.join(CACHE_DIR, f"{model_variant_str}_{BEST_MODEL_NAME}")
print(f"--- Configuration: {model_variant_str} ---")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

if USE_NEWS_FEATURES:
    PROCESSED_NEWS_CACHE_PATH = os.path.join(CACHE_DIR, "all_processed_news_feature_new10days.pt")
    FORCE_RECOMPUTE_NEWS = False
else:
    PROCESSED_NEWS_CACHE_PATH = None
    FORCE_RECOMPUTE_NEWS = False

# --- TimeXer-Specific Configurations ---
class TimeXerConfigs:
    """A configuration class for the TimeXer model's specific needs."""
    def __init__(self, num_coins, price_seq_len, pred_len=1, d_model=64, freq='h',
                embed_type='timeF', patch_len=16, stride=8, e_layers=1,
                d_ff=256, dropout=0.1, fc_dropout=0.1, head_dropout=0.1,
                act='gelu', n_heads=4, factor=5, use_norm=False, task_type='regression'):
        self.enc_in = num_coins # 编码器输入特征维度 (即节点数)
        self.dec_in = num_coins # 解码器输入特征维度 (在本模型中与编码器相同)
        self.c_out = num_coins # 输出特征维度 (即节点数)
        self.seq_len = price_seq_len # 输入序列长度
        self.pred_len = pred_len # 预测长度 (本模型预测下一时间步)
        self.d_model = d_model # 模型的主要特征维度 (patch embedding后的维度)
        self.freq = freq # 时间序列频率 (如 'h' 小时)
        self.embed = embed_type # 时间特征编码类型 (如 'timeF', 'fixed', 'learned')
        self.patch_len = patch_len # Patch的长度
        self.stride = stride # Patch的滑动步长
        self.d_ff = d_ff # 前馈网络的隐藏层维度
        self.e_layers = e_layers # 编码器的层数
        self.dropout = dropout # Dropout比例
        self.fc_dropout = fc_dropout # 前馈网络的Dropout比例
        self.head_dropout = head_dropout # 注意力头的Dropout比例
        self.factor = factor # Attention相关参数 (通常用于自注意力)
        self.activation = act # 激活函数
        self.n_heads = n_heads # Attention头的数量
        self.use_norm = use_norm # 是否使用归一化层
        self.task_type = task_type

def save_test_predictions(all_preds, all_targets, coin_names, timestamp=None):
    """
    保存测试集的预测值和真实值到CSV文件

    Args:
        all_preds: 预测值数组 [num_samples, num_coins]
        all_targets: 真实值数组 [num_samples, num_coins]
        coin_names: 币种名称列表
        timestamp: 时间戳字符串，如果为None则自动生成
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建保存目录
    save_dir = "experiments/cache/test_predictions"
    os.makedirs(save_dir, exist_ok=True)

    # 保存详细预测结果
    predictions_file = os.path.join(save_dir, f"test_predictions_{timestamp}.csv")
    with open(predictions_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'coin', 'true_value', 'predicted_value', 'absolute_error', 'percentage_error'])

        for sample_idx in range(len(all_preds)):
            for coin_idx, coin_name in enumerate(coin_names):
                true_val = all_targets[sample_idx, coin_idx]
                pred_val = all_preds[sample_idx, coin_idx]
                abs_error = abs(true_val - pred_val)
                pct_error = (abs_error / abs(true_val)) * 100 if abs(true_val) > 1e-8 else float('inf')

                writer.writerow([sample_idx, coin_name, true_val, pred_val, abs_error, pct_error])

    # 保存统计信息
    statistics_file = os.path.join(save_dir, f"test_statistics_{timestamp}.csv")
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

    print(f"测试集预测结果已保存到:")
    print(f"  详细结果: {predictions_file}")
    print(f"  统计信息: {statistics_file}")

    return predictions_file, statistics_file

def evaluate_model(model, data_loader, criterion, edge_index, device, task_type, scaler=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating"):
            price_seq = batch_data['price_seq'].to(device)
            price_seq_mark = batch_data['price_seq_mark'].to(device)
            target_data = batch_data['target_price'].to(device)
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(device)

            x_enc = price_seq # Prepare input for TimeXer
            outputs = model(x_enc, price_seq_mark, edge_index=edge_index, news_features=news_features)

            if task_type == 'classification':
                targets = (target_data > 0).long()
                loss = criterion(outputs.view(-1, NUM_CLASSES), targets.view(-1))
                preds = torch.argmax(outputs, dim=-1)
            else: # regression
                if PREDICTION_TARGET == 'price':
                    targets = target_data
                elif PREDICTION_TARGET == 'diff':
                    last_price_in_seq = price_seq[:, -1, :]
                    targets = target_data - last_price_in_seq
                else:
                    raise ValueError(f"Unknown PREDICTION_TARGET: {PREDICTION_TARGET}")
                loss = criterion(outputs, targets)
                preds = outputs

            total_loss += loss.item() * price_seq.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = {'loss': avg_loss}
    if task_type == 'classification':
        accuracy = (all_preds == all_targets).sum() / all_targets.size
        metrics['accuracy'] = accuracy
        coin_accuracies = {}
        for i, coin_name in enumerate(COIN_NAMES):
            coin_correct = (all_preds[:, i] == all_targets[:, i]).sum()
            coin_total = len(all_targets[:, i])
            coin_accuracies[coin_name] = coin_correct / coin_total if coin_total > 0 else 0
        metrics['coin_accuracies'] = coin_accuracies
    else: # regression
        # Denormalize for metrics calculation based on the prediction target
        if PREDICTION_TARGET == 'price':
            if scaler:
                num_coins = all_preds.shape[1]
                original_preds = scaler.inverse_transform(all_preds.reshape(-1, num_coins))
                original_targets = scaler.inverse_transform(all_targets.reshape(-1, num_coins))
            else:
                original_preds = all_preds
                original_targets = all_targets
        elif PREDICTION_TARGET == 'diff':
            if scaler:
                num_coins = all_preds.shape[1]
                preds_flat = all_preds.reshape(-1, num_coins)
                targets_flat = all_targets.reshape(-1, num_coins)
                if isinstance(scaler, StandardScaler):
                    original_preds = preds_flat * scaler.scale_
                    original_targets = targets_flat * scaler.scale_
                elif isinstance(scaler, MinMaxScaler):
                    data_range = scaler.data_max_ - scaler.data_min_
                    original_preds = preds_flat * data_range
                    original_targets = targets_flat * data_range
                else:
                    original_preds = preds_flat
                    original_targets = targets_flat
            else:
                original_preds = all_preds
                original_targets = all_targets
        else:
            raise ValueError(f"Unknown PREDICTION_TARGET: {PREDICTION_TARGET}")

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
            
            # Calculate normalized errors (divided by mean price)
            norm_targets = coin_targets / coin_mean
            norm_preds = coin_preds / coin_mean
            
            # Calculate metrics
            mae = mean_absolute_error(norm_targets, norm_preds)  # Normalized MAE
            mse = mean_squared_error(norm_targets, norm_preds)   # Normalized MSE
            rmse = np.sqrt(mse)                                  # Normalized RMSE
            mape = mean_absolute_percentage_error(coin_targets, coin_preds)  # MAPE (already relative)
            r2 = r2_score(coin_targets, coin_preds)             # R² (already scale-invariant)
            
            # Store metrics for averaging
            coin_maes.append(mae)
            coin_mses.append(mse)
            coin_mapes.append(mape)
            coin_r2s.append(r2)
            
            # Store per-coin metrics
            per_coin_metrics[coin_name] = {
                'normalized_mae': mae,
                'normalized_mse': mse,
                'normalized_rmse': rmse,
                'mape': mape,
                'r2': r2,
                # Also store original metrics for reference
                'mae': mean_absolute_error(coin_targets, coin_preds),
                'mse': mean_squared_error(coin_targets, coin_preds),
                'rmse': np.sqrt(mean_squared_error(coin_targets, coin_preds))
            }
        
        # Calculate new MAE: sum of all true values / sum of all predicted values
        total_true_sum = np.sum(original_targets)
        total_pred_sum = np.sum(original_preds)
        new_mae = total_true_sum / total_pred_sum if total_pred_sum != 0 else float('inf')

        # Calculate both original and normalized overall metrics
        metrics.update({
            # Original metrics (calculated directly on all data)
            'mae': mean_absolute_error(original_targets, original_preds),  # 原来的MAE计算方式
            'new_mae': new_mae,  # 新的MAE计算方式：所有真实值之和除以预测值之和
            'mse': mean_squared_error(original_targets, original_preds),
            'rmse': np.sqrt(mean_squared_error(original_targets, original_preds)),
            'r2': r2_score(original_targets, original_preds),
            'mape': mean_absolute_percentage_error(original_targets, original_preds),

            # Normalized metrics (average of per-coin metrics)
            'normalized_mae': np.mean(coin_maes),      # Average of normalized MAEs
            'normalized_mse': np.mean(coin_mses),      # Average of normalized MSEs
            'normalized_rmse': np.sqrt(np.mean(coin_mses)),  # RMSE of normalized errors
            'avg_mape': np.mean(coin_mapes),          # Average MAPE
            'avg_r2': np.mean(coin_r2s),             # Average R²
            'median_mape': np.median(coin_mapes),     # Median MAPE
            'worst_mape': np.max(coin_mapes),         # Worst MAPE
            'best_mape': np.min(coin_mapes)           # Best MAPE
        })
        
        metrics['per_coin_metrics'] = per_coin_metrics

    return metrics, all_preds, all_targets

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Print configuration summary
    print(f"\n=== Configuration Summary ===")
    print(f"Task Type: {TASK_TYPE}")
    print(f"Use News: {USE_NEWS_FEATURES}")
    print(f"Use GCN: {USE_GCN}")
    print(f"Normalization: {NORM_TYPE}")
    print(f"Sequence Length: {PRICE_SEQ_LEN}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Model: TimeXer (d_model={D_MODEL}, patch_len={PATCH_LEN}, stride={STRIDE})")
    print(f"================================\n")

    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    edge_index = generate_edge_index(price_df_full, THRESHOLD).to(DEVICE) if USE_GCN else None

    fit_train_size = int(len(price_df_full) * (1 - VALIDATION_SPLIT_RATIO - TEST_SPLIT_RATIO))
    price_df_for_scaler = price_df_full.iloc[:fit_train_size]
    
    if NORM_TYPE == 'standard': scaler = StandardScaler()
    elif NORM_TYPE == 'minmax': scaler = MinMaxScaler()
    else: scaler = None

    if scaler:
        scaler.fit(price_df_for_scaler)
        price_df_values = scaler.transform(price_df_full)
        price_df = pd.DataFrame(price_df_values, columns=price_df_full.columns, index=price_df_full.index)
    else:
        price_df = price_df_full

    news_data_dict = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) if USE_NEWS_FEATURES else None
    
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        news_data_dict=news_data_dict,
        seq_len=PRICE_SEQ_LEN,
        processed_news_features_path=PROCESSED_NEWS_CACHE_PATH,
        force_recompute_news=FORCE_RECOMPUTE_NEWS,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET, # Must be true for TimeXer
        time_freq=TIME_FREQ_IN_DATASET
    )
    
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    timexer_configs = TimeXerConfigs(
        num_coins=len(COIN_NAMES), 
        price_seq_len=PRICE_SEQ_LEN,
        d_model=D_MODEL,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        e_layers=E_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT_TIMEXER,
        task_type=TASK_TYPE
    )
    
    model = UnifiedTimexerGCN(
        configs=timexer_configs,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        use_gcn=USE_GCN,
        news_feature_dim=dataset.news_feature_dim if USE_NEWS_FEATURES else None,
        news_processed_dim=NEWS_PROCESSED_DIM,
        mlp_hidden_dim_1=MLP_HIDDEN_DIM_1,
        mlp_hidden_dim_2=MLP_HIDDEN_DIM_2,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    print(model)

    if TASK_TYPE == 'classification':
        criterion = nn.CrossEntropyLoss()
    else: # regression
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7, min_lr=1e-6)
    
    best_val_metric = float('inf')
    patience_counter = 0
    early_stopping_patience = 10  # 早停耐心值
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")
        for batch_data in train_pbar:
            price_seq = batch_data['price_seq'].to(DEVICE)
            price_seq_mark = batch_data['price_seq_mark'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(DEVICE)

            optimizer.zero_grad()
            
            x_enc = price_seq # Prepare input for TimeXer
            outputs = model(x_enc, price_seq_mark, edge_index=edge_index, news_features=news_features)
            
            if TASK_TYPE == 'classification':
                targets = (target_data > 0).long()
                loss = criterion(outputs.view(-1, NUM_CLASSES), targets.view(-1))
            else: # regression
                if PREDICTION_TARGET == 'price':
                    targets = target_data
                elif PREDICTION_TARGET == 'diff':
                    last_price_in_seq = price_seq[:, -1, :]
                    targets = target_data - last_price_in_seq
                else:
                    raise ValueError(f"Unknown PREDICTION_TARGET: {PREDICTION_TARGET}")
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * price_seq.size(0)
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_dataset)
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, edge_index, DEVICE, TASK_TYPE, scaler)

        val_loss = val_metrics['loss']
        scheduler.step(val_loss)

        print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("--- Validation Metrics (Overall) ---")
        for metric_name, value in val_metrics.items():
            if not isinstance(value, dict):
                print(f"  - {metric_name.upper()}: {value:.4f}")

        if val_loss < best_val_metric:
            best_val_metric = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🚀 New best model saved to {BEST_MODEL_PATH} (Val Loss: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

    print("\n--- Starting Testing with Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("Warning: Best model not found. Testing with the last state.")

    test_metrics, test_preds, test_targets = evaluate_model(model, test_loader, criterion, edge_index, DEVICE, TASK_TYPE, scaler)

    # 保存测试集预测结果到新文件夹
    if TASK_TYPE == 'regression':
        # 对于回归任务，需要反归一化预测值和真实值用于保存
        if PREDICTION_TARGET == 'price' and scaler:
            num_coins = test_preds.shape[1]
            original_test_preds = scaler.inverse_transform(test_preds.reshape(-1, num_coins))
            original_test_targets = scaler.inverse_transform(test_targets.reshape(-1, num_coins))
        else:
            original_test_preds = test_preds
            original_test_targets = test_targets

        save_test_predictions(original_test_preds, original_test_targets, COIN_NAMES)

    print(f"\n✅ Test Results:")
    print("  Overall:")
    for metric_name, value in test_metrics.items():
        if not isinstance(value, dict):
            print(f"    - {metric_name.upper()}: {value:.4f}")

    if 'coin_accuracies' in test_metrics:
        print("\n  --- Per-Coin Accuracy (Test) ---")
        acc_strings = [f"{coin}: {acc:.4f}" for coin, acc in test_metrics['coin_accuracies'].items()]
        print("    " + " | ".join(acc_strings))

    if 'per_coin_metrics' in test_metrics:
        print("\n  --- Per-Coin Regression Metrics (Test) ---")
        for coin_name, coin_metrics in test_metrics['per_coin_metrics'].items():
            print(f"  --- {coin_name} ---")
            for metric_name, value in coin_metrics.items():
                print(f"    - {metric_name.upper()}: {value:.4f}")
    print("\n--- Script Finished ---") 