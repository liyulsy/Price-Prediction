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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import the unified model and dataset
from models.MixModel.unified_cnn_gnn import UnifiedCnnGnn
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)


# --- Configuration and Hyperparameters ---
# Master switches to control the model architecture and task
TASK_TYPE = 'regression'
# This script is for price prediction, so GCN and News are disabled.
USE_NEWS_FEATURES = False
USE_GCN = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Paths ---
PRICE_CSV_PATH = 'crypto_analysis/data/processed_data/1H/all_1H.csv'
CACHE_DIR = "cache"
BEST_MODEL_NAME = "best_cnn_model.pt"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 60
NORM_TYPE = 'standard' # Using standard scaler for prices

# --- Model Parameters ---
NEWS_PROCESSED_DIM = 32
CNN_OUTPUT_CHANNELS = 64
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 128
FINAL_MLP_HIDDEN_DIM = 256
NUM_CLASSES = 2 # Not used in regression

# --- Training Parameters ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 20

# --- Dynamic parameter based on the master switch ---
model_variant = ['CNN', TASK_TYPE]
if USE_NEWS_FEATURES: model_variant.append("with_news")
else: model_variant.append("no_news")
if USE_GCN: model_variant.append("with_gcn")
else: model_variant.append("no_gcn")

model_variant_str = "_".join(model_variant)
BEST_MODEL_PATH = os.path.join(CACHE_DIR, f"{model_variant_str}_{BEST_MODEL_NAME}")
print(f"--- Configuration: {model_variant_str} ---")
print(f"Best model will be saved to: {BEST_MODEL_PATH}")

def mean_absolute_percentage_error(y_true, y_pred):
    """A robust MAPE calculation that handles potential division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(model, data_loader, criterion, device, scaler=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating", leave=False):
            price_seq = batch_data['price_seq'].to(device)
            target_data = batch_data['target_price'].to(device)
            
            # Model call is simplified, no edge_index or news_features
            outputs = model(price_seq, edge_index=None, news_features=None)

            targets = target_data
            loss = criterion(outputs, targets)
            preds = outputs

            total_loss += loss.item() * price_seq.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = {'loss': avg_loss}

    # Denormalize for metrics calculation
    if scaler:
        # Use standard inverse_transform for absolute prices
        original_preds = scaler.inverse_transform(all_preds)
        original_targets = scaler.inverse_transform(all_targets)
    else:
        original_preds = all_preds
        original_targets = all_targets
        
    # Calculate metrics on original scale
    metrics['mae'] = mean_absolute_error(original_targets, original_preds)
    metrics['mse'] = mean_squared_error(original_targets, original_preds)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(original_targets, original_preds)
    metrics['mape'] = mean_absolute_percentage_error(original_targets, original_preds)

    # Per-coin regression metrics
    per_coin_metrics = {}
    for i, coin_name in enumerate(COIN_NAMES):
        coin_targets = original_targets[:, i]
        coin_preds = original_preds[:, i]
        
        per_coin_metrics[coin_name] = {
            'mse': mean_squared_error(coin_targets, coin_preds),
            'rmse': np.sqrt(mean_squared_error(coin_targets, coin_preds)),
            'mae': mean_absolute_error(coin_targets, coin_preds),
            'r2': r2_score(coin_targets, coin_preds),
            'mape': mean_absolute_percentage_error(coin_targets, coin_preds)
        }
    
    metrics['per_coin_metrics'] = per_coin_metrics

    return metrics

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
    print(f"================================\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 1. Load and process price data
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    # The merged data might already have the correct column names (e.g., 'BTC-USDT')
    # but the model internals use the base name (e.g., 'BTC').
    # We create a rename map to be safe.
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]
    
    # 2. Normalize price data
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

    # 3. Create Dataset and DataLoaders using the original UnifiedCryptoDataset
    # This model doesn't use time encoding from the dataset.
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        seq_len=PRICE_SEQ_LEN,
        news_data_dict=None,
        time_encoding_enabled=False
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
    
    # 4. Initialize Unified Model
    model = UnifiedCnnGnn(
        price_seq_len=PRICE_SEQ_LEN,
        num_nodes=len(COIN_NAMES),
        task_type=TASK_TYPE,
        use_gcn=USE_GCN,
        news_feature_dim=None,
        news_processed_dim=NEWS_PROCESSED_DIM,
        cnn_output_channels=CNN_OUTPUT_CHANNELS,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    print(model)

    # Loss function is MSE for price prediction
    criterion = nn.MSELoss()

    # Initialize optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
            targets = batch_data['target_price'].to(DEVICE)

            optimizer.zero_grad()
            # Model call is simplified
            outputs = model(price_seq, edge_index=None, news_features=None)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            
            optimizer.step()
            epoch_loss += loss.item() * price_seq.size(0)
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_loss / len(train_dataset)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, scaler)
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("--- Validation Metrics (Overall) ---")
        for metric_name, value in val_metrics.items():
            if not isinstance(value, dict):
                print(f"  - {metric_name.upper()}: {value:.4f}")

        # Early stopping check and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🚀 New best model saved to {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epochs")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            break

    # 7. Testing
    print("\n--- Starting Testing with Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    else:
        print("Warning: Best model not found. Testing with the last state.")

    test_metrics = evaluate_model(model, test_loader, criterion, DEVICE, scaler)
    print(f"\n✅ Test Results:")
    # Print overall metrics
    print("  Overall:")
    for metric_name, value in test_metrics.items():
        if not isinstance(value, dict):
            print(f"    - {metric_name.upper()}: {value:.4f}")

    # Print per-coin metrics for the final test results
    if 'per_coin_metrics' in test_metrics:
        print("\n  --- Per-Coin Regression Metrics (Test) ---")
        for coin_name, coin_metrics in test_metrics['per_coin_metrics'].items():
            print(f"  --- {coin_name} ---")
            for metric_name, value in coin_metrics.items():
                print(f"    - {metric_name.upper()}: {value:.4f}")
    print("\n--- Script Finished ---") 