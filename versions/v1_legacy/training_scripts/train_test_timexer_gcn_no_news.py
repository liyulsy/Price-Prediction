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

# Model and Dataset Imports
from models.MixModel.timexer_gcn_no_news import TimexerGCN # 修改导入
from scripts.analysis.crypto_new_analyzer.dataset_no_news import CryptoDatasetNoNews # load_news_data 仍然保留导入，但不会调用
from dataloader.gnn_loader import generate_edge_index

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)


# --- TimeXer Configuration Class ---
class TimeXerConfigs:
    # TimeXer模型配置参数
    def __init__(self, num_nodes, price_seq_len, num_time_features, # 节点数，价格序列输入长度，时间特征数量
                 d_model=64, pred_len=1, label_len_ratio=0.5, # 模型主要特征维度，预测长度，标签长度比例
                 dropout=0.1, n_heads=4, d_ff=128, e_layers=2, factor=5, # Dropout比例，Attention头数量，前馈网络维度，编码器层数，Attention相关参数
                 patch_len=12, stride=6, freq='h', output_attention=False, embed_type='timeF', # Patch长度，Patch滑动步长，时间序列频率，是否输出注意力权重，时间特征编码类型
                 use_norm: bool = False): # 是否使用归一化层
        self.enc_in = num_nodes # 编码器输入特征维度 (即节点数)
        self.dec_in = num_nodes # 解码器输入特征维度 (在本模型中与编码器相同)
        self.c_out = num_nodes # 输出特征维度 (即节点数)
        self.d_model = d_model # 模型的主要特征维度 (patch embedding后的维度)
        self.seq_len = price_seq_len # 输入序列长度
        self.pred_len = pred_len # 预测长度 (本模型预测下一时间步)
        self.label_len = int(price_seq_len * label_len_ratio) # 标签长度，用于TimeXer的Decoder (本模型未使用Decoder)
        self.output_attention = output_attention # 是否输出注意力权重 (本模型未使用Decoder)
        self.embed = embed_type # 时间特征编码类型 (如 'timeF', 'fixed', 'learned')
        self.freq = freq # 时间序列频率 (如 'h' 小时)
        self.dropout = dropout # Dropout比例
        self.n_heads = n_heads # Attention头的数量
        self.d_ff = d_ff # 前馈网络的隐藏层维度
        self.factor = factor # Attention相关参数 (通常用于自注意力)
        self.activation = 'gelu' # 激活函数
        self.e_layers = e_layers # 编码器的层数
        self.patch_len = patch_len # Patch的长度
        self.stride = stride # Patch的滑动步长
        self.use_norm = use_norm # 是否使用归一化层
        self.num_time_features = num_time_features # 时间特征的数量

# --- Main Configuration and Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 使用的设备 (GPU或CPU)

# Data paths
PRICE_CSV_PATH = 'datafiles/price_data/1H.csv' # 原始价格数据CSV文件路径
# NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features' # 移除新闻特征文件夹路径
# PROCESSED_NEWS_CACHE_PATH = "cache/all_processed_news_features.pt" # 移除处理后的新闻特征缓存文件路径

# Dataset parameters
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX'] # 加载的币种名称列表 (节点)
PRICE_SEQ_LEN = 60 # 价格时间序列输入长度
THRESHOLD = 0.6 # 构建图时边的阈值 (相关性阈值)
NORM_TYPE = 'standard' # 价格数据归一化类型 ('standard', 'minmax', 'none')
TIME_ENCODING_ENABLED_IN_DATASET = True # 数据集中是否启用时间编码
TIME_FREQ_IN_DATASET = 'h' # 数据集使用的时间频率 (用于时间编码)

# TimeXerConfigs parameters (to be instantiated later after dataset)
D_MODEL = 128 # TimeXer模型主要特征维度
PATCH_LEN = 24 # TimeXer Patch长度
STRIDE = 6 # TimeXer Patch滑动步长
E_LAYERS = 3 # TimeXer编码器层数
N_HEADS = 4 # TimeXer Attention头数量
D_FF = 64 # TimeXer 前馈网络隐藏层维度
DROPOUT_TIMEXER = 0.3 # TimeXer Dropout比例

# TimexerGCN specific model parameters
# NEWS_PROCESSED_DIM = 32 # 移除处理后的新闻特征维度
GCN_HIDDEN_DIM = 128 # GCN隐藏层维度
GCN_OUTPUT_DIM = 64 # GCN输出维度


# Training parameters
BATCH_SIZE = 16 # 训练批量大小
EPOCHS = 20 # 总训练轮数
LEARNING_RATE = 0.0005 # 学习率
WEIGHT_DECAY = 1e-5 # 权重衰减 (L2正则化)
VALIDATION_SPLIT_RATIO = 0.15 # 验证集划分比例
TEST_SPLIT_RATIO = 0.15 # 测试集划分比例
FORCE_RECOMPUTE_NEWS = False # 是否强制重新计算并缓存新闻特征

BEST_MODEL_PATH = "Project1/cache/best_timexer_gcn_no_news_model_v2.pt" # 最佳模型保存路径

def evaluate_model(model, data_loader, criterion, edge_index, device):
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
            x_mark_enc = batch_data['price_seq_mark'].to(device)
            target_prices = batch_data['target_price'].to(device)

            target_labels = (target_prices > 0).long()
            
            outputs = model(price_seq, x_mark_enc, edge_index)
            
            loss = criterion(outputs.view(-1, model.mlp[-1].out_features), target_labels.view(-1))
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

    # --- 1. Load and Preprocess Price Data (same as before) --- # 加载和预处理价格数据
    print("\n--- 1. Loading and Preprocessing Price Data ---\n")
    if not os.path.exists(PRICE_CSV_PATH):
        print(f"错误: 价格数据文件未找到于 {PRICE_CSV_PATH}")
        exit()
    # if not os.path.exists(NEWS_FEATURES_FOLDER): # 移除新闻特征文件夹检查
    #     print(f"错误: 新闻特征文件夹未找到于 {NEWS_FEATURES_FOLDER}")
    #     exit()
        
    price_df_original_load = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    expected_csv_columns = [f"{coin}-USDT" for coin in COIN_NAMES]
    missing_cols = [col for col in expected_csv_columns if col not in price_df_original_load.columns]
    if missing_cols:
        print(f"错误: CSV '{PRICE_CSV_PATH}' 缺少列: {missing_cols}")
        exit()
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_processed_cols = price_df_original_load.rename(columns=rename_map)
    price_df_final_cols = price_df_processed_cols[COIN_NAMES]

    # --- 2. Define edge_index (same as before) --- # 定义图的边索引
    print("\n--- 2. Defining Edge Index ---\n")
    edge_index = generate_edge_index(price_df_final_cols, THRESHOLD).to(DEVICE)

    # --- 3. Data Normalization (same as before) --- # 数据归一化
    print(f"\n--- 3. Applying Data Normalization (Type: {NORM_TYPE}) ---\n")
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

    # --- 4. Create CryptoDataset and DataLoaders --- # 创建数据集和DataLoader
    print("\n--- 4. Creating Datasets and DataLoaders ---\n")
    # news_data = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) # 移除新闻数据加载
    news_data = None # 将新闻数据设为 None

    dataset = CryptoDatasetNoNews(
        price_data_df=price_df,
        seq_len=PRICE_SEQ_LEN,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET,
        time_freq=TIME_FREQ_IN_DATASET
    )
    NUM_NODES = dataset.num_coins
    # NEWS_FEATURE_DIM will be 0 if news_data_dict is None, which is correct for the no news model
    # NEWS_FEATURE_DIM = dataset.news_feature_dim # 保留以获取为 0 的维度
    ACTUAL_NUM_TIME_FEATURES = dataset.num_actual_time_features
    print(f"Dataset created. Nodes: {NUM_NODES}, TimeFeatDim: {ACTUAL_NUM_TIME_FEATURES}, Samples: {len(dataset)}")
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

    # --- 5. Initialize Model, Loss, Optimizer --- # 初始化模型、损失函数和优化器
    print("\n--- 5. Initializing Model, Loss, Optimizer ---\n")
    
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
        # news_feature_dim=NEWS_FEATURE_DIM, # 移除新闻特征维度参数
        # news_processed_dim=NEWS_PROCESSED_DIM, # 移除处理后的新闻特征维度参数
    ).to(DEVICE)
    print(model)

    # 2. 加载参数
    # model.load_state_dict(torch.load("cache/hpo_timexer/trial_2_best_model.pt", map_location=DEVICE)) # 默认不加载预训练参数
    # print(f"已加载参数文件: cache/hpo_timexer/trial_2_best_model.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # --- 6. Training Loop --- # 训练循环
    print("\n--- 6. Starting Training ---\n")
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
            target_prices = batch_data['target_price'].to(DEVICE)

            target_labels = (target_prices > 0).long()
            
            optimizer.zero_grad()
            outputs = model(price_seq, x_mark_enc, edge_index)
            
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
        
        val_loss, val_accuracy, val_coin_accuracies = evaluate_model(model, val_loader, criterion, edge_index, DEVICE)
        
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
    
    # --- 7. Testing Step --- # 测试步骤
    print("\n--- 7. Starting Testing with Best Model ---\n")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded best model from {BEST_MODEL_PATH}")
    else:
        print(f"Warning: Best model {BEST_MODEL_PATH} not found. Testing with last model state.")

    test_loss, test_accuracy, test_coin_accuracies = evaluate_model(model, test_loader, criterion, edge_index, DEVICE)
    print(f"\n✅ Test Results:")
    print(f"Overall - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
    print("Coin-wise Accuracies:")
    for coin, acc in test_coin_accuracies.items():
        print(f"  {coin}: {acc:.4f}")

    print("\n--- Script Finished ---") 