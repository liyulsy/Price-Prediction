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

# 导入模型和数据集
from models.MixModel.lstm_gcn import LstmGcn
from scripts.analysis.crypto_new_analyzer.dataset import CryptoDataset  # 支持新闻的数据集
from scripts.analysis.crypto_new_analyzer.dataset_no_news import CryptoDatasetNoNews  # 不支持新闻的数据集
from dataloader.gnn_loader import generate_edge_index
from scripts.analysis.crypto_new_analyzer.dataset import load_news_data

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)


# --- 配置和超参数 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
PRICE_CSV_PATH = 'datafiles/price_data/1H.csv'
NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features'  # 新闻特征文件夹路径
PROCESSED_NEWS_CACHE_PATH = "cache/all_processed_news_feature_10days.pt"  # 预处理新闻特征的缓存路径

# 数据集参数
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 60  # LSTM的输入序列长度
THRESHOLD = 0.6  # 用于生成邻接矩阵的阈值
NORM_TYPE = 'standard'  # 'standard', 'minmax', or 'none'

# LSTM-GCN模型参数
INPUT_DIM = 1          # 每个时间步的特征维度（这里假设只用收盘价）
LSTM_HIDDEN_DIM = 64   # LSTM隐藏层维度
LSTM_OUT_DIM = 32      # LSTM输出维度
NEWS_PROCESSED_DIM = 32 # 处理后的新闻特征维度
GCN_HIDDEN_DIM = 64    # GCN隐藏层维度
GCN_OUTPUT_DIM = 32    # GCN输出维度
MLP_HIDDEN_DIM = 128   # MLP隐藏层维度
NUM_CLASSES = 2        # 分类任务的类别数（涨/跌）
USE_GCN = False        # 是否使用GCN
USE_NEWS = False        # 是否使用新闻特征
FORCE_RECOMPUTE_NEWS = False  # 是否强制重新计算新闻特征

# 训练参数
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15

# 模型保存路径
BEST_MODEL_PATH = "cache/best_lstm_gcn_model.pt"

def evaluate_model(model, data_loader, criterion, edge_index, device, use_news=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 为每个币种创建统计变量
    coin_correct = {coin: 0 for coin in COIN_NAMES}
    coin_total = {coin: 0 for coin in COIN_NAMES}
    
    all_targets_flat = []
    all_preds_flat = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating"):
            price_seq = batch_data['price_seq'].to(device)
            target_prices = batch_data['target_price'].to(device)
            target_labels = (target_prices > 0).long()

            # 根据是否使用新闻特征决定forward参数
            if use_news:
                news_features = batch_data['news_features'].to(device)
                outputs = model(price_seq, edge_index, news_features)
            else:
                outputs = model(price_seq, edge_index)
            
            loss = criterion(outputs.view(-1, NUM_CLASSES), target_labels.view(-1))
            total_loss += loss.item() * price_seq.size(0)
            
            preds = torch.argmax(outputs, dim=-1)
            total_correct += (preds == target_labels).sum().item()
            total_samples += target_labels.numel()

            # 计算每个币种的正确数和样本数
            for i, coin in enumerate(COIN_NAMES):
                coin_correct[coin] += (preds[:, i] == target_labels[:, i]).sum().item()
                coin_total[coin] += preds[:, i].numel()

            all_targets_flat.extend(target_labels.view(-1).cpu().numpy())
            all_preds_flat.extend(preds.view(-1).cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # 计算每个币种的准确率
    coin_accuracies = {coin: coin_correct[coin] / coin_total[coin] if coin_total[coin] > 0 else 0 
                      for coin in COIN_NAMES}
    
    return avg_loss, accuracy, coin_accuracies

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # --- 1. 加载和预处理数据 ---
    print("\n--- 1. Loading and Preprocessing Data ---")
    if not os.path.exists(PRICE_CSV_PATH):
        print(f"错误: 价格数据文件未找到于 {PRICE_CSV_PATH}")
        exit()
    
    if USE_NEWS and not os.path.exists(NEWS_FEATURES_FOLDER):
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

    # --- 2. 定义edge_index ---
    print("\n--- 2. Defining Edge Index ---")
    edge_index = generate_edge_index(price_df_final_cols, THRESHOLD).to(DEVICE)

    # --- 3. 数据归一化 ---
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
    else:
        price_df = price_df_to_normalize
        if NORM_TYPE != 'none':
            print(f"警告: 未知的NORM_TYPE '{NORM_TYPE}'。未对价格数据进行归一化处理。")

    # --- 4. 创建数据集和DataLoader ---
    print("\n--- 4. Creating Datasets and DataLoaders ---")
    if USE_NEWS:
        print("使用带新闻特征的数据集")
        news_data = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES)
        dataset = CryptoDataset(
            price_data_df=price_df,
            news_data_dict=news_data,
            seq_len=PRICE_SEQ_LEN,
            processed_news_features_path=PROCESSED_NEWS_CACHE_PATH,
            force_recompute_news=FORCE_RECOMPUTE_NEWS
        )
        NEWS_FEATURE_DIM = dataset.news_feature_dim
        print(f"新闻特征维度: {NEWS_FEATURE_DIM}")
    else:
        print("使用不带新闻特征的数据集")
        dataset = CryptoDatasetNoNews(
            price_data_df=price_df,
            seq_len=PRICE_SEQ_LEN
        )
        NEWS_FEATURE_DIM = None

    NUM_NODES = dataset.num_coins
    print(f"Dataset created. Nodes: {NUM_NODES}, Samples: {len(dataset)}")
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

    # --- 5. 初始化模型、损失函数、优化器 ---
    print("\n--- 5. Initializing Model, Loss, Optimizer ---")
    model = LstmGcn(
        seq_len=PRICE_SEQ_LEN,
        num_nodes=NUM_NODES,
        input_dim=INPUT_DIM,
        news_feature_dim=NEWS_FEATURE_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_out_dim=LSTM_OUT_DIM,
        news_processed_dim=NEWS_PROCESSED_DIM,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        use_gcn=USE_GCN,
        use_news=USE_NEWS
    ).to(DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # --- 6. 训练循环 ---
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
            target_prices = batch_data['target_price'].to(DEVICE)
            target_labels = (target_prices > 0).long()
            
            optimizer.zero_grad()
            
            # 根据是否使用新闻特征决定forward参数
            if USE_NEWS:
                news_features = batch_data['news_features'].to(DEVICE)
                outputs = model(price_seq, edge_index, news_features)
            else:
                outputs = model(price_seq, edge_index)
            
            loss = criterion(outputs.view(-1, NUM_CLASSES), target_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * price_seq.size(0)
            
            preds = torch.argmax(outputs, dim=-1)
            epoch_train_correct += (preds == target_labels).sum().item()
            epoch_train_samples += target_labels.numel()
            
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_accuracy = epoch_train_correct / epoch_train_samples if epoch_train_samples > 0 else 0
        
        val_loss, val_accuracy, val_coin_accuracies = evaluate_model(model, val_loader, criterion, edge_index, DEVICE, USE_NEWS)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}")
        print("Val Coin Accuracies:")
        for coin, acc in val_coin_accuracies.items():
            print(f"  {coin}: {acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
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
    
    # --- 7. 测试步骤 ---
    print("\n--- 7. Starting Testing with Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded best model from {BEST_MODEL_PATH}")
    else:
        print(f"Warning: Best model {BEST_MODEL_PATH} not found. Testing with last model state.")

    test_loss, test_accuracy, test_coin_accuracies = evaluate_model(model, test_loader, criterion, edge_index, DEVICE, USE_NEWS)
    print(f"\n✅ Test Results:")
    print(f"Overall - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    print("Coin-wise Accuracies:")
    for coin, acc in test_coin_accuracies.items():
        print(f"  {coin}: {acc:.4f}")

    print("\n--- Script Finished ---")