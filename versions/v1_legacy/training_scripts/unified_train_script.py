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

# Import the unified model and dataset
from models.MixModel.unified_cnn_gnn import UnifiedCnnGnn
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
BEST_MODEL_NAME = "best_unified_model.pt"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 60  # 增加序列长度以捕捉更长期依赖
THRESHOLD = 0.6
NORM_TYPE = 'standard'

# --- Model Parameters ---
NEWS_PROCESSED_DIM = 32
CNN_OUTPUT_CHANNELS = 64  # 增加通道数
GCN_HIDDEN_DIM = 256  # 增加GCN隐藏层维度
GCN_OUTPUT_DIM = 128  # 增加GCN输出维度
FINAL_MLP_HIDDEN_DIM = 256  # 增加MLP隐藏层维度
NUM_CLASSES = 2

# --- Training Parameters ---
BATCH_SIZE = 32  # 增大批次大小
EPOCHS = 20  # 增加训练轮数
LEARNING_RATE = 0.001  # 初始学习率
MIN_LR = 1e-6  # 最小学习率
WEIGHT_DECAY = 1e-4  # 增加权重衰减
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# --- Learning Rate Schedule Parameters ---
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-6
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# --- Gradient Clipping ---
MAX_GRAD_NORM = 1.0

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 1e-4

# --- Dynamic parameter based on the master switch ---
model_variant = [TASK_TYPE]
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
            target_data = batch_data['target_price'].to(device)
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(device)
            
            outputs = model(price_seq, edge_index, news_features=news_features)

            if task_type == 'classification':
                # Classification target is ALWAYS based on price difference,
                # regardless of the regression prediction target.
                last_price_in_seq = price_seq[:, -1, :]
                price_diff = target_data - last_price_in_seq
                targets = (price_diff > 0).long()
                loss = criterion(outputs.view(-1, NUM_CLASSES), targets.view(-1))
                preds = torch.argmax(outputs, dim=-1)
            else: # regression
                if PREDICTION_TARGET == 'price':
                    # Target is the next time step's absolute price
                    targets = target_data
                elif PREDICTION_TARGET == 'diff':
                    # Target is the difference between next price and current price
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
        # Per-coin accuracy
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
                # Use standard inverse_transform for absolute prices
                original_preds = scaler.inverse_transform(all_preds)
                original_targets = scaler.inverse_transform(all_targets)
            else:
                original_preds = all_preds
                original_targets = all_targets
        elif PREDICTION_TARGET == 'diff':
            if scaler:
                num_coins = all_preds.shape[1]
                preds_flat = all_preds.reshape(-1, num_coins)
                targets_flat = all_targets.reshape(-1, num_coins)

                if isinstance(scaler, StandardScaler):
                    # For price difference, only multiply by scale_ (std)
                    original_preds = preds_flat * scaler.scale_
                    original_targets = targets_flat * scaler.scale_
                elif isinstance(scaler, MinMaxScaler):
                    # For price difference, only multiply by data range
                    data_range = scaler.data_max_ - scaler.data_min_
                    original_preds = preds_flat * data_range
                    original_targets = targets_flat * data_range
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
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 1. Load and process price data
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    # 2. Generate edge_index
    edge_index = generate_edge_index(price_df_full, THRESHOLD).to(DEVICE) if USE_GCN else None

    # 3. Normalize price data
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

    # 4. Create UnifiedCryptoDataset and DataLoaders
    news_data_dict = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) if USE_NEWS_FEATURES else None
    
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        news_data_dict=news_data_dict,
        seq_len=PRICE_SEQ_LEN,
        processed_news_features_path=PROCESSED_NEWS_CACHE_PATH,
        force_recompute_news=FORCE_RECOMPUTE_NEWS,
        # Pass predict_mode if needed for inference script, here we are training
    )
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Initialize Unified Model
    model = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN,
        num_nodes=dataset.num_coins,
        task_type=TASK_TYPE,
        use_gcn=USE_GCN,
        news_feature_dim=dataset.news_feature_dim if USE_NEWS_FEATURES else None,
        news_processed_dim=NEWS_PROCESSED_DIM,
        cnn_output_channels=CNN_OUTPUT_CHANNELS,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    print(model)

    # Choose loss function based on task
    if TASK_TYPE == 'classification':
        criterion = nn.CrossEntropyLoss()
    else: # regression
        criterion = nn.MSELoss()

    # Initialize optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=WARMUP_START_LR, weight_decay=WEIGHT_DECAY)
    
    # Learning rate schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=WARMUP_START_LR/LEARNING_RATE,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )
    
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        min_lr=MIN_LR
    )
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training")
        
        for batch_data in train_pbar:
            price_seq = batch_data['price_seq'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(price_seq, edge_index, news_features=news_features)
            
            if TASK_TYPE == 'classification':
                # Classification target is ALWAYS based on price difference,
                # regardless of the regression prediction target.
                last_price_in_seq = price_seq[:, -1, :]
                price_diff = target_data - last_price_in_seq
                targets = (price_diff > 0).long()
                loss = criterion(outputs.view(-1, NUM_CLASSES), targets.view(-1))
            else: # regression
                if PREDICTION_TARGET == 'price':
                    # Target is the next time step's absolute price
                    targets = target_data
                elif PREDICTION_TARGET == 'diff':
                    # Target is the difference between next price and current price
                    last_price_in_seq = price_seq[:, -1, :]
                    targets = target_data - last_price_in_seq
                else:
                    raise ValueError(f"Unknown PREDICTION_TARGET: {PREDICTION_TARGET}")

                loss = criterion(outputs, targets)

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            epoch_loss += loss.item() * price_seq.size(0)
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_dataset)
        
        # Validation
        val_metrics, _, _ = evaluate_model(model, val_loader, criterion, edge_index, DEVICE, TASK_TYPE, scaler)
        val_loss = val_metrics['loss']
        
        # Learning rate scheduling
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            main_scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🚀 New best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Print epoch summary
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        if current_lr != optimizer.param_groups[0]['lr']:
            print(f"Learning rate decreased to: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Early Stopping Counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Print detailed metrics
        print("\nValidation Metrics:")
        for metric_name, value in val_metrics.items():
            if not isinstance(value, dict):
                print(f"  - {metric_name.upper()}: {value:.4f}")

    # 7. Testing
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
            original_test_preds = scaler.inverse_transform(test_preds)
            original_test_targets = scaler.inverse_transform(test_targets)
        else:
            original_test_preds = test_preds
            original_test_targets = test_targets

        save_test_predictions(original_test_preds, original_test_targets, COIN_NAMES)

    print(f"\n✅ Test Results:")
    # Print overall metrics
    print("  Overall:")
    for metric_name, value in test_metrics.items():
        if not isinstance(value, dict):
            print(f"    - {metric_name.upper()}: {value:.4f}")

    # Print per-coin metrics for the final test results
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