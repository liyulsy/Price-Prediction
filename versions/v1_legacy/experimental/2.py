import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Model and Dataset Imports
from models.MixModel.timexer_gcn_with_lstm import TimexerGCN
from crypto_new_analyzer.dataset import CryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index

# --- TimeXer Configuration Class ---
class TimeXerConfigs:
    def __init__(self, num_nodes, price_seq_len, num_time_features,
                 d_model=64, pred_len=1, label_len_ratio=0.5, 
                 dropout=0.1, n_heads=4, d_ff=128, e_layers=2, factor=5,
                 patch_len=12, stride=6, freq='h', output_attention=False, embed_type='timeF',
                 use_norm: bool = False):
        self.enc_in = num_nodes
        self.dec_in = num_nodes
        self.c_out = num_nodes
        self.d_model = d_model
        self.seq_len = price_seq_len
        self.pred_len = pred_len 
        self.label_len = int(price_seq_len * label_len_ratio)
        self.output_attention = output_attention
        self.embed = embed_type
        self.freq = freq
        self.dropout = dropout
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.factor = factor
        self.activation = 'gelu'
        self.e_layers = e_layers
        self.patch_len = patch_len
        self.stride = stride
        self.use_norm = use_norm
        self.num_time_features = num_time_features

# --- Main Configuration and Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
PRICE_CSV_PATH = 'datafiles/price_data/1H.csv'
NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features'
PROCESSED_NEWS_CACHE_PATH = "cache/all_processed_news_feature_10days.pt"

# Dataset parameters
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 60
THRESHOLD = 0.6
NORM_TYPE = 'standard'
TIME_ENCODING_ENABLED_IN_DATASET = True
TIME_FREQ_IN_DATASET = 'h'

# TimeXerConfigs parameters (to be instantiated later after dataset)
D_MODEL = 64
PATCH_LEN = 24
STRIDE = 12
E_LAYERS = 2
N_HEADS = 4
D_FF = 128
DROPOUT_TIMEXER = 0.1

# TimexerGCN specific model parameters
NEWS_PROCESSED_DIM = 32
GCN_HIDDEN_DIM = 128
GCN_OUTPUT_DIM = 64
MODEL_DROPOUT = 0.3

# Training parameters
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
FORCE_RECOMPUTE_NEWS = False

BEST_MODEL_PATH = "Project1/cache/best_timexer_gcn_with_lstm_model_v2.pt"

def evaluate_model(model, data_loader, criterion, edge_index, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_targets_flat = []
    all_preds_flat = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating"):
            price_seq = batch_data['price_seq'].to(device)
            x_mark_enc = batch_data['price_seq_mark'].to(device)
            news_features = batch_data['news_features'].to(device)
            target_prices = batch_data['target_price'].to(device)

            target_labels = (target_prices > 0).long()
            
            outputs = model(price_seq, x_mark_enc, edge_index, news_features)
            
            loss = criterion(outputs.view(-1, model.mlp[-1].out_features), target_labels.view(-1))
            total_loss += loss.item() * price_seq.size(0)
            
            preds = torch.argmax(outputs, dim=-1)
            total_correct += (preds == target_labels).sum().item()
            total_samples += target_labels.numel()

            all_targets_flat.extend(target_labels.view(-1).cpu().numpy())
            all_preds_flat.extend(preds.view(-1).cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # --- 1. Load and Preprocess Price Data (same as before) ---
    print("\n--- 1. Loading and Preprocessing Price Data ---")
    if not os.path.exists(PRICE_CSV_PATH):
        print(f"错误: 价格数据文件未找到于 {PRICE_CSV_PATH}")
        exit()
    if not os.path.exists(NEWS_FEATURES_FOLDER):
        print(f"错误: 新闻特征文件夹未找到于 {NEWS_FEATURES_FOLDER}")
        exit()
        
    price_df_original_load = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    expected_csv_columns = [f"{coin}-USDT" for coin in COIN_NAMES]
    missing_cols = [col for col in expected_csv_columns if col not in price_df_original_load.columns]
    if missing_cols:
        print(f"错误: CSV '{PRICE_CSV_PATH}' 缺少列: {missing_cols}")
        exit()
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_processed_cols = price_df_original_load.rename(columns=rename_map)
    price_df_final_cols = price_df_processed_cols[COIN_NAMES]

    # --- 2. Define edge_index (same as before) ---
    print("\n--- 2. Defining Edge Index ---")
    edge_index = generate_edge_index(price_df_final_cols, THRESHOLD).to(DEVICE)

    # --- 3. Data Normalization (same as before) ---
    print(f"\n--- 3. Applying Data Normalization (Type: {NORM_TYPE}) ---")
    num_total_samples = len(price_df_final_cols)
    fit_train_size = int(num_total_samples * (1 - VALIDATION_SPLIT_RATIO - TEST_SPLIT_RATIO))
    if fit_train_size <= 0:
        print(f"错误: 数据集太小或划分比例不当. Fit train size: {fit_train_size}")
        exit()
    price_df_for_scaler_fit = price_df_final_cols.iloc[:fit_train_size]
    price_df_to_normalize = price_df_final_cols.copy()
    if NORM_TYPE == 'standard':
        scaler = StandardScaler()
        price_df_values_full = scaler.fit_transform(price_df_to_normalize)
        price_df = pd.DataFrame(price_df_values_full, columns=price_df_to_normalize.columns, index=price_df_to_normalize.index)
    elif NORM_TYPE == 'minmax':
        scaler = MinMaxScaler()
        price_df_values_full = scaler.fit_transform(price_df_to_normalize)
        price_df = pd.DataFrame(price_df_values_full, columns=price_df_to_normalize.columns, index=price_df_to_normalize.index)
    elif NORM_TYPE == 'none':
        price_df = price_df_to_normalize
    else:
        price_df = price_df_to_normalize
        print(f"警告: 未知的NORM_TYPE '{NORM_TYPE}'。未对价格数据进行归一化处理。")

    # --- 4. Create CryptoDataset and DataLoaders ---
    print("\n--- 4. Creating Datasets and DataLoaders ---")
    news_data = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES)
    
    dataset = CryptoDataset(
        price_data_df=price_df,
        news_data_dict=news_data,
        seq_len=PRICE_SEQ_LEN,
        processed_news_features_path=PROCESSED_NEWS_CACHE_PATH,
        force_recompute_news=FORCE_RECOMPUTE_NEWS,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET,
        time_freq=TIME_FREQ_IN_DATASET
    )
    NUM_NODES = dataset.num_coins
    NEWS_FEATURE_DIM = dataset.news_feature_dim
    ACTUAL_NUM_TIME_FEATURES = dataset.num_actual_time_features
    print(f"Dataset created. Nodes: {NUM_NODES}, NewsFeatDim: {NEWS_FEATURE_DIM}, TimeFeatDim: {ACTUAL_NUM_TIME_FEATURES}, Samples: {len(dataset)}")
    if len(dataset) == 0:
        print("错误: 数据集为空。")
        exit()

    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        print(f"错误: 数据集太小无法划分. Train: {train_size}, Val: {val_size}, Test: {test_size}")
        exit()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 5. Initialize Model, Loss, Optimizer ---
    print("\n--- 5. Initializing Model, Loss, Optimizer ---")
    
    timexer_model_configs = TimeXerConfigs(
        num_nodes=NUM_NODES,
        price_seq_len=PRICE_SEQ_LEN,
        num_time_features=ACTUAL_NUM_TIME_FEATURES,
        d_model=D_MODEL,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        e_layers=E_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT_TIMEXER,
        freq=TIME_FREQ_IN_DATASET
    )
    print(f"TimeXer Configs: enc_in={timexer_model_configs.enc_in}, seq_len={timexer_model_configs.seq_len}, d_model={timexer_model_configs.d_model}, num_time_feat={timexer_model_configs.num_time_features}")

    model = TimexerGCN(
        configs=timexer_model_configs,
        hidden_dim=GCN_HIDDEN_DIM,
        output_dim=GCN_OUTPUT_DIM,
        news_feature_dim=NEWS_FEATURE_DIM,
        news_processed_dim=NEWS_PROCESSED_DIM,
    ).to(DEVICE)
    print(model)

    # # 2. 加载参数
    # model.load_state_dict(torch.load("cache/hpo_timexer/trial_2_best_model.pt", map_location=DEVICE))
    # print(f"已加载参数文件: cache/hpo_timexer/trial_2_best_model.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # --- 6. Training Loop ---
    print("\n--- 6. Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 10

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)
        for batch_data in train_pbar:
            price_seq = batch_data['price_seq'].to(DEVICE)
            x_mark_enc = batch_data['price_seq_mark'].to(DEVICE)
            news_features = batch_data['news_features'].to(DEVICE)
            target_prices = batch_data['target_price'].to(DEVICE)

            target_labels = (target_prices > 0).long()
            
            optimizer.zero_grad()
            outputs = model(price_seq, x_mark_enc, edge_index, news_features)
            
            loss = criterion(outputs.view(-1, model.mlp[-1].out_features), target_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * price_seq.size(0)
            
            preds = torch.argmax(outputs, dim=-1)
            epoch_train_correct += (preds == target_labels).sum().item()
            epoch_train_samples += target_labels.numel()
            
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_accuracy = epoch_train_correct / epoch_train_samples if epoch_train_samples > 0 else 0
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, edge_index, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if BEST_MODEL_PATH and not os.path.exists(os.path.dirname(BEST_MODEL_PATH)) and os.path.dirname(BEST_MODEL_PATH) != '':
                os.makedirs(os.path.dirname(BEST_MODEL_PATH))
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🚀 New best model saved to {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stopping:
                print(f"⏳ Early stopping after {patience_early_stopping} epochs with no improvement.")
                break
    
    # --- 7. Testing Step ---
    print("\n--- 7. Starting Testing with Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded best model from {BEST_MODEL_PATH}")
    else:
        print(f"Warning: Best model {BEST_MODEL_PATH} not found. Testing with last model state.")

    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, edge_index, DEVICE)
    print(f"✅ Test Results - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    print("\n--- Script Finished ---") 