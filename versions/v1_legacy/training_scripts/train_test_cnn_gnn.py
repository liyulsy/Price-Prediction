import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import os
import sys
import numpy as np # For accuracy calculation and other metrics if needed
from sklearn.preprocessing import StandardScaler, MinMaxScaler # 导入归一化工具


# 假设模型和数据集代码在以下路径 (请根据您的项目结构调整)
from models.MixModel.cnn_gnn import CnnGnn
from scripts.analysis.crypto_new_analyzer.dataset import CryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# --- 配置和超参数 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据路径
PRICE_CSV_PATH = 'datafiles/price_data/1H.csv' # 请确保这是正确的价格数据文件路径
NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features' # 新闻特征文件夹路径
PROCESSED_NEWS_CACHE_PATH = "cache/new_all_processed_news_feature_10days.pt" # 预处理新闻特征的缓存路径

# 数据集参数
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX'] # 您的币种列表
PRICE_SEQ_LEN = 60 # CNN的输入序列长度 (例如，过去180个时间点)
THRESHOLD = 0.6 # 用于生成邻接矩阵的阈值
NORM_TYPE = 'standard' # 新增: 'standard', 'minmax', or 'none'
# NUM_NODES 将从 COIN_NAMES 长度动态获取
# NEWS_FEATURE_DIM 将从 CryptoDataset 实例动态获取

# CnnGnn 模型参数
NEWS_PROCESSED_DIM = 64       # 新闻特征经过MLP处理后的维度
CNN_OUTPUT_CHANNELS = 32     # CNN输出维度
GCN_HIDDEN_DIM = 128          # GCN隐藏层维度
GCN_OUTPUT_DIM = 64           # GCN输出层维度 (也是最终MLP的输入部分)
FINAL_MLP_HIDDEN_DIM = 128    # 最终MLP的隐藏层维度
NUM_CLASSES = 2               # 分类任务的类别数量 (例如，2表示涨/跌)

# 训练参数
BATCH_SIZE = 16               # 批量大小
EPOCHS = 20                   # 训练周期数
LEARNING_RATE = 0.0005        # 学习率
WEIGHT_DECAY = 1e-5           # 优化器的权重衰减
VALIDATION_SPLIT_RATIO = 0.15 # 从总数据集中分出作为验证集的比例
TEST_SPLIT_RATIO = 0.15       # 从总数据集中分出作为测试集的比例
FORCE_RECOMPUTE_NEWS = True  # 是否强制重新计算新闻特征

# 用于保存最佳模型的路径
BEST_MODEL_PATH = "cache/best_cnn_gnn_model.pt"

def evaluate_model(model, data_loader, criterion, edge_index, num_nodes, device):
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
            news_features = batch_data['news_features'].to(device)
            target_prices = batch_data['target_price'].to(device)

            target_labels = (target_prices > 0).long()
            
            outputs = model(price_seq, edge_index, news_features)
            
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
    if not os.path.exists(NEWS_FEATURES_FOLDER):
        print(f"错误: 新闻特征文件夹未找到于 {NEWS_FEATURES_FOLDER}")
        exit()
        
    price_df_original_load = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    
    # 列名处理和排序 (已有逻辑)
    print(f"原始CSV加载的列名: {price_df_original_load.columns.tolist()}")
    expected_csv_columns = [f"{coin}-USDT" for coin in COIN_NAMES]
    missing_cols = [col for col in expected_csv_columns if col not in price_df_original_load.columns]
    if missing_cols:
        print(f"错误: 价格数据CSV文件 '{PRICE_CSV_PATH}' 中缺少以下预期的列: {missing_cols}")
        print(f"基于COIN_NAMES = {COIN_NAMES}, 脚本期望找到列: {expected_csv_columns}")
        exit()
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_processed_cols = price_df_original_load.rename(columns=rename_map)
    price_df_final_cols = price_df_processed_cols[COIN_NAMES]
    print(f"列名处理和排序后的DataFrame列: {price_df_final_cols.columns.tolist()}")

    # --- 2. 定义 edge_index (使用未归一化的、但已整理好列的数据) ---
    print("\n--- 2. Defining Edge Index (using original scale data) ---")
    # 注意: generate_edge_index 可能需要pd.DataFrame作为输入，确保price_df_final_cols是DataFrame
    edge_index = generate_edge_index(price_df_final_cols, THRESHOLD).to(DEVICE)
    print(f"Edge index created (shape: {edge_index.shape})")

    # --- 3. 数据归一化/标准化 ---
    print(f"\n--- 3. Applying Data Normalization (Type: {NORM_TYPE}) ---")
    
    # 确定用于拟合scaler的训练数据部分的索引范围
    # 我们用整个数据集的前 (1 - VALIDATION_SPLIT_RATIO - TEST_SPLIT_RATIO) 部分来拟合scaler
    # 这假设数据是按时间排序的，并且未来的数据不会影响过去的scaler参数
    num_total_samples = len(price_df_final_cols)
    # 确保这里的划分比例与后续DataLoader的划分逻辑意图上大致对应
    # 但请注意，random_split 是随机的，而这里的scaler拟合部分是固定的头部数据
    fit_train_size = int(num_total_samples * (1 - VALIDATION_SPLIT_RATIO - TEST_SPLIT_RATIO))
    
    if fit_train_size <= 0:
        print(f"错误: 数据集太小 (或划分比例不当)，无法划定有效的训练部分来拟合归一化scaler。计算得到的拟合用样本数: {fit_train_size}")
        exit()
        
    price_df_for_scaler_fit = price_df_final_cols.iloc[:fit_train_size]

    price_df_to_normalize = price_df_final_cols.copy() # 将要进行归一化的数据

    if NORM_TYPE == 'standard':
        scaler = StandardScaler()
        scaler.fit(price_df_for_scaler_fit) # 在训练数据部分拟合
        price_df_values = scaler.transform(price_df_to_normalize) # 对整个数据集转换
        price_df = pd.DataFrame(price_df_values, columns=price_df_to_normalize.columns, index=price_df_to_normalize.index)
        print("价格数据已进行标准化处理。")
    elif NORM_TYPE == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(price_df_for_scaler_fit) # 在训练数据部分拟合
        price_df_values = scaler.transform(price_df_to_normalize) # 对整个数据集转换
        price_df = pd.DataFrame(price_df_values, columns=price_df_to_normalize.columns, index=price_df_to_normalize.index)
        print("价格数据已进行Min-Max归一化处理。")
    elif NORM_TYPE == 'none':
        price_df = price_df_to_normalize # 未归一化，直接使用整理好列的数据
        print("未对价格数据进行归一化处理。")
    else:
        price_df = price_df_to_normalize
        print(f"警告: 未知的NORM_TYPE '{NORM_TYPE}'。未对价格数据进行归一化处理。")
    
    # --- 4. 创建 CryptoDataset 和 DataLoader ---
    print("\n--- 4. Creating Datasets and DataLoaders ---")
    news_data = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES)
    dataset = CryptoDataset(
        price_data_df=price_df, # 此处 price_df 是可能已归一化的
        news_data_dict=news_data,
        seq_len=PRICE_SEQ_LEN,
        processed_news_features_path=PROCESSED_NEWS_CACHE_PATH,
        force_recompute_news=FORCE_RECOMPUTE_NEWS
    )
    NUM_NODES = dataset.num_coins
    NEWS_FEATURE_DIM = dataset.news_feature_dim
    print(f"Dataset created. Number of nodes: {NUM_NODES}, News feature dim: {NEWS_FEATURE_DIM}, Total samples: {len(dataset)}")
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查数据文件和预处理步骤。")
        exit()

    # 划分数据集 (使用可能已归一化的数据创建的dataset)
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        print(f"错误: 数据集太小无法按指定比例划分。Train: {train_size}, Val: {val_size}, Test: {test_size}")
        exit()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Train dataset size: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- 5. 初始化模型、损失函数、优化器 ---
    # 注意：edge_index 已提前生成并移至DEVICE
    print("\n--- 5. Initializing Model, Loss, Optimizer ---")
    model = CnnGnn(
        price_seq_len=PRICE_SEQ_LEN,
        num_nodes=NUM_NODES,
        news_feature_dim=NEWS_FEATURE_DIM,
        news_processed_dim=NEWS_PROCESSED_DIM,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_output_dim=GCN_OUTPUT_DIM,
        cnn_output_channels=CNN_OUTPUT_CHANNELS,
        final_mlp_hidden_dim=FINAL_MLP_HIDDEN_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # 调整patience

    # --- 6. 训练循环 ---
    print("\n--- 6. Starting Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience_early_stopping = 10 # 如果验证损失连续10个周期没有改善则早停

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False)
        for batch_data in train_pbar:
            price_seq = batch_data['price_seq'].to(DEVICE)
            news_features = batch_data['news_features'].to(DEVICE)
            target_prices = batch_data['target_price'].to(DEVICE)

            target_labels = (target_prices > 0).long()
            
            optimizer.zero_grad()
            outputs = model(price_seq, edge_index, news_features) # Shape: [B, N, NumClasses]
            
            loss = criterion(outputs.view(-1, NUM_CLASSES), target_labels.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * price_seq.size(0) # batch loss
            
            preds = torch.argmax(outputs, dim=-1)
            epoch_train_correct += (preds == target_labels).sum().item()
            epoch_train_samples += target_labels.numel()
            
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / len(train_dataset)
        train_accuracy = epoch_train_correct / epoch_train_samples if epoch_train_samples > 0 else 0
        
        # 验证步骤
        val_loss, val_accuracy, val_coin_accuracies = evaluate_model(model, val_loader, criterion, edge_index, NUM_NODES, DEVICE)
        
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
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🚀 New best model saved to {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stopping:
                print(f"⏳ Early stopping triggered after {patience_early_stopping} epochs with no improvement on validation loss.")
                break
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")


    # --- 7. 测试步骤 ---
    print("\n--- 7. Starting Testing with Best Model ---")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded best model from {BEST_MODEL_PATH} for testing.")
    else:
        print(f"Warning: Best model file {BEST_MODEL_PATH} not found. Testing with the last state of the model.")

    test_loss, test_accuracy, test_coin_accuracies = evaluate_model(model, test_loader, criterion, edge_index, NUM_NODES, DEVICE)
    print(f"\n✅ Test Results:")
    print(f"Overall - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    print("Coin-wise Accuracies:")
    for coin, acc in test_coin_accuracies.items():
        print(f"  {coin}: {acc:.4f}")

    print("\n--- Script Finished ---") 