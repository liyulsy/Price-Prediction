import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Import the unified Timexer model and dataset components
from models.MixModel.unified_timexer_gcn import UnifiedTimexerGCN
from crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index
from train_timexer import TimeXerConfigs, evaluate_model

# --- Configuration and Hyperparameters ---
# Master switches to control the model architecture and task
TASK_TYPE = 'regression'
PREDICTION_TARGET = 'price'
USE_NEWS_FEATURES = False
USE_GCN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Paths ---
PRICE_CSV_PATH = 'crypto_analysis/data/processed_data/1H/all_1H.csv'
NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features'
CACHE_DIR = "cache"
BEST_MODEL_NAME = "best_timexer_model.pt"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 90  # 贝叶斯优化得到的最佳序列长度
THRESHOLD = 0.6
NORM_TYPE = 'minmax'  # 改为 MinMaxScaler，更适合回归任务
TIME_ENCODING_ENABLED_IN_DATASET = True
TIME_FREQ_IN_DATASET = 'h'

# --- Model Parameters (贝叶斯优化得到的最佳参数) ---
NEWS_PROCESSED_DIM = 64
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 128
MLP_HIDDEN_DIM_1 = 512
MLP_HIDDEN_DIM_2 = 128
NUM_CLASSES = 2

D_MODEL = 256
PATCH_LEN = 24
STRIDE = 4
E_LAYERS = 3
N_HEADS = 4
D_FF = 512
DROPOUT_TIMEXER = 0.15

# --- Testing Parameters ---
BATCH_SIZE = 32
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

def format_float(x):
    return f"{x:.4f}" if isinstance(x, (float, np.float64)) else str(x)

def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Print configuration summary
    print(f"\n=== Configuration Summary ===")
    print(f"Task Type: {TASK_TYPE}")
    print(f"Use News: {USE_NEWS_FEATURES}")
    print(f"Use GCN: {USE_GCN}")
    print(f"Normalization: {NORM_TYPE}")
    print(f"Sequence Length: {PRICE_SEQ_LEN}")
    print(f"Model: TimeXer")
    print(f"  - d_model: {D_MODEL}")
    print(f"  - patch_len: {PATCH_LEN}")
    print(f"  - stride: {STRIDE}")
    print(f"  - e_layers: {E_LAYERS}")
    print(f"  - n_heads: {N_HEADS}")
    print(f"  - d_ff: {D_FF}")
    print(f"  - dropout: {DROPOUT_TIMEXER}")
    print(f"================================\n")

    # 1. Load and process data
    print("Loading data...")
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    edge_index = generate_edge_index(price_df_full, THRESHOLD).to(DEVICE) if USE_GCN else None

    # 2. Normalize data
    if NORM_TYPE == 'standard': scaler = StandardScaler()
    elif NORM_TYPE == 'minmax': scaler = MinMaxScaler()
    else: scaler = None

    if scaler:
        print(f"Normalizing data using {NORM_TYPE} scaler...")
        price_df_values = scaler.fit_transform(price_df_full)
        price_df = pd.DataFrame(price_df_values, columns=price_df_full.columns, index=price_df_full.index)
    else:
        price_df = price_df_full

    # 3. Create dataset
    print("Creating dataset...")
    news_data_dict = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) if USE_NEWS_FEATURES else None
    
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        news_data_dict=news_data_dict,
        seq_len=PRICE_SEQ_LEN,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET,
        time_freq=TIME_FREQ_IN_DATASET
    )
    
    # 4. Split dataset
    print("Splitting dataset...")
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Initialize model with optimized parameters
    print("Initializing model...")
    timexer_configs = TimeXerConfigs(
        num_coins=len(COIN_NAMES),
        price_seq_len=PRICE_SEQ_LEN,
        task_type=TASK_TYPE,
        d_model=D_MODEL,
        e_layers=E_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT_TIMEXER,
        freq=TIME_FREQ_IN_DATASET,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        embed_type='timeF',  # 默认使用时间特征编码
        pred_len=1,  # 预测下一个时间步
        fc_dropout=0.1,  # 默认值
        head_dropout=0.1,  # 默认值
        factor=5,  # 默认值
        use_norm=False  # 默认值
    )

    model = UnifiedTimexerGCN(
        configs=timexer_configs,
        use_gcn=USE_GCN,
        news_feature_dim=dataset.news_feature_dim if USE_NEWS_FEATURES else None,
        news_processed_dim=NEWS_PROCESSED_DIM,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        mlp_hidden_dim_1=MLP_HIDDEN_DIM_1,
        mlp_hidden_dim_2=MLP_HIDDEN_DIM_2,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # 6. Load best model if exists
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading best model from {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    else:
        print("Warning: Best model not found!")

    # 7. Testing
    print("\n--- Starting Testing ---")
    criterion = nn.CrossEntropyLoss() if TASK_TYPE == 'classification' else nn.MSELoss()
    test_metrics = evaluate_model(model, test_loader, criterion, edge_index, DEVICE, TASK_TYPE, scaler)

    # 8. Print results in table format
    print("\n=== Test Results ===")
    
    # Overall metrics table
    overall_metrics = {k: v for k, v in test_metrics.items() if not isinstance(v, dict)}
    overall_df = pd.DataFrame({
        'Metric': list(overall_metrics.keys()),
        'Value': [format_float(v) for v in overall_metrics.values()]
    })
    print("\nOverall Metrics:")
    print(overall_df.to_string(index=False))

    # Per-coin metrics table
    if 'per_coin_metrics' in test_metrics:
        print("\nPer-Coin Metrics:")
        per_coin_data = []
        for coin, metrics in test_metrics['per_coin_metrics'].items():
            metrics['Coin'] = coin
            # Format all float values
            metrics = {k: format_float(v) for k, v in metrics.items()}
            per_coin_data.append(metrics)
        
        per_coin_df = pd.DataFrame(per_coin_data)
        cols = ['Coin'] + [col for col in per_coin_df.columns if col != 'Coin']
        per_coin_df = per_coin_df[cols]
        print(per_coin_df.to_string(index=False))

    print("\n--- Testing Finished ---")

if __name__ == '__main__':
    main() 