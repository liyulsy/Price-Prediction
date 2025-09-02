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
import optuna
import json

# Model and Dataset Imports
from models.MixModel.timexer_gcn import TimexerGCN
from scripts.analysis.crypto_new_analyzer.dataset import CryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)


# --- TimeXer Configuration Class ---
class TimeXerConfigs:
    # TimeXer模型配置参数
    def __init__(self, num_nodes, price_seq_len, num_time_features,
                 d_model=64, pred_len=1, label_len_ratio=0.5, 
                 dropout=0.1, n_heads=4, d_ff=128, e_layers=2, factor=5,
                 patch_len=12, stride=6, freq='h', output_attention=False, embed_type='timeF',
                 use_norm: bool = False):
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
NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features' # 新闻特征文件夹路径
PROCESSED_NEWS_GLOBAL_CACHE_PATH = "cache/all_processed_news_feature_10days.pt" # 处理后的新闻特征缓存文件路径
BEST_MODEL_PATH_TEMPLATE = "cache/hpo_timexer/trial_{}_best_model.pt" # 最佳模型保存路径模板 (用于HPO)

# Dataset parameters
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX'] # 加载的币种名称列表 (节点)
PRICE_SEQ_LEN = 180 # 价格时间序列输入长度
THRESHOLD = 0.6 # 构建图时边的阈值 (相关性阈值)
NORM_TYPE = 'standard' # 价格数据归一化类型 ('standard', 'minmax', 'none')
TIME_ENCODING_ENABLED_IN_DATASET = True # 数据集中是否启用时间编码
TIME_FREQ_IN_DATASET = 'h' # 数据集使用的时间频率 (用于时间编码)

# TimeXerConfigs parameters (to be instantiated later after dataset)
# 这些是TimeXer的默认配置，在HPO时会被覆盖
D_MODEL = 64 
PATCH_LEN = 24
STRIDE = 12
E_LAYERS = 2
N_HEADS = 4
D_FF = 128
DROPOUT_TIMEXER = 0.1

# TimexerGCN specific model parameters
NEWS_PROCESSED_DIM = 32 # 处理后的新闻特征维度
GCN_HIDDEN_DIM = 128 # GCN隐藏层维度
GCN_OUTPUT_DIM = 64 # GCN输出维度
MODEL_DROPOUT = 0.3 # TimexerGCN模型整体Dropout比例 ( currently not explicitly used in TimexerGCN class itself, but good to note)

# Training parameters
BATCH_SIZE = 16 # 训练批量大小
EPOCHS = 20 # 总训练轮数 (用于非HPO模式，或在HPO外运行)
LEARNING_RATE = 0.0005 # 学习率
WEIGHT_DECAY = 1e-5 # 权重衰减 (L2正则化)
VALIDATION_SPLIT_RATIO = 0.15 # 验证集划分比例
TEST_SPLIT_RATIO = 0.15 # 测试集划分比例
FORCE_RECOMPUTE_NEWS_GLOBAL = False # 是否强制重新计算并缓存新闻特征

BEST_MODEL_PATH = "cache/best_timexer_gcn_model_v2.pt" # 最佳模型保存路径 (用于非HPO模式)

# HPO specific settings
EPOCHS_PER_TRIAL = 15 # 每个Optuna trial训练的轮数
N_WARMUP_STEPS_PRUNING = 5 # Optuna剪枝前热身步数

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

def objective(trial):
    print(f"\n--- Starting Optuna Trial {trial.number} ---")
    
    # Ensure cache directories exist ONLY for model saving per trial
    # The news cache is global and its directory should exist or be created once outside
    os.makedirs(os.path.dirname(BEST_MODEL_PATH_TEMPLATE.format(trial.number)), exist_ok=True) # 确保保存模型的目录存在

    # --- 1. Hyperparameter Sampling ---
    # Dataset/Preprocessing HPs
    price_seq_len_hp = trial.suggest_int('price_seq_len', 60, 240, step=30) # 价格序列输入长度采样
    scheduler_patience_hp = trial.suggest_int('scheduler_patience', 3, 7) # 学习率调度器耐心值采样

    # TimeXer相关参数采样
    d_model_hp = trial.suggest_categorical('d_model', [32, 48, 64, 96, 128]) # TimeXer模型主要特征维度采样
    patch_len_hp = trial.suggest_categorical('patch_len', [6, 12, 24, 36, 48]) # TimeXer Patch长度采样
    stride_hp = trial.suggest_categorical('stride', [3, 6, 12, 18, 24]) # TimeXer Patch滑动步长采样
    e_layers_hp = trial.suggest_int('e_layers', 1, 4) # TimeXer编码器层数采样
    n_heads_hp = trial.suggest_categorical('n_heads', [2, 4, 6, 8]) # TimeXer Attention头数量采样
    d_ff_hp = trial.suggest_categorical('d_ff', [32, 64, 128, 192, 256]) # TimeXer 前馈网络隐藏层维度采样
    dropout_hp = trial.suggest_float('dropout', 0.05, 0.5, step=0.05) # TimeXer Dropout比例采样

    print(f"[Trial {trial.number}] TimeXer参数: d_model={d_model_hp}, patch_len={patch_len_hp}, stride={stride_hp}, e_layers={e_layers_hp}, n_heads={n_heads_hp}, d_ff={d_ff_hp}, dropout={dropout_hp}") # 打印采样的TimeXer参数
    
    # --- 2. Data Loading and Preprocessing ---
    # Load raw price data (could be done once outside objective if it's large and static)
    if not os.path.exists(PRICE_CSV_PATH):
        print(f"错误: 价格数据文件未找到于 {PRICE_CSV_PATH}") # 检查价格数据文件是否存在
        exit()
    if not os.path.exists(NEWS_FEATURES_FOLDER):
        print(f"错误: 新闻特征文件夹未找到于 {NEWS_FEATURES_FOLDER}") # 检查新闻特征文件夹是否存在
        exit()
        
    price_df_original_load = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True) # 读取原始价格数据
    expected_csv_columns = [f"{coin}-USDT" for coin in COIN_NAMES] # 预期的CSV列名
    missing_cols = [col for col in expected_csv_columns if col not in price_df_original_load.columns] # 检查是否存在缺失列
    if missing_cols:
        print(f"错误: CSV '{PRICE_CSV_PATH}' 缺少列: {missing_cols}") # 打印缺失列信息并退出
        exit()
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES} # 构建列名映射
    price_df_processed_cols = price_df_original_load.rename(columns=rename_map) # 重命名列
    price_df_final_cols = price_df_processed_cols[COIN_NAMES] # 选择目标币种列

    # --- 3. Data Normalization (same as before) ---
    print(f"\n--- 3. Applying Data Normalization (Type: {NORM_TYPE}) ---") # 打印归一化类型信息
    num_total_samples = len(price_df_final_cols) # 总样本数
    fit_train_size = int(num_total_samples * (1 - VALIDATION_SPLIT_RATIO - TEST_SPLIT_RATIO)) # 计算用于fit scaler的训练集大小
    if fit_train_size <= 0:
        print(f"错误: 数据集太小或划分比例不当. Fit train size: {fit_train_size}") # 检查数据集大小是否足够
        exit()
    price_df_for_scaler_fit = price_df_final_cols.iloc[:fit_train_size] # 用于fit scaler的数据
    price_df_to_normalize = price_df_final_cols.copy() # 待归一化的数据副本
    if NORM_TYPE == 'standard':
        scaler = StandardScaler() # 标准化Scaler
        price_df_values_full = scaler.fit_transform(price_df_to_normalize) # Fit并转换
        price_df_normalized = pd.DataFrame(price_df_values_full, columns=price_df_to_normalize.columns, index=price_df_to_normalize.index) # 转换为DataFrame
    elif NORM_TYPE == 'minmax':
        scaler = MinMaxScaler() # MinMaxScaler
        price_df_values_full = scaler.fit_transform(price_df_to_normalize) # Fit并转换
        price_df_normalized = pd.DataFrame(price_df_values_full, columns=price_df_to_normalize.columns, index=price_df_to_normalize.index) # 转换为DataFrame
    elif NORM_TYPE == 'none':
        price_df_normalized = price_df_to_normalize # 不进行归一化
    else:
        price_df_normalized = price_df_to_normalize
        print(f"警告: 未知的NORM_TYPE '{NORM_TYPE}'。未对价格数据进行归一化处理。") # 未知归一化类型警告

    news_data = load_news_data(NEWS_FEATURES_FOLDER, COIN_NAMES) # 加载新闻数据
    
    dataset = CryptoDataset(
        price_data_df=price_df_normalized, # 归一化后的价格数据
        news_data_dict=news_data, # 新闻数据字典
        seq_len=price_seq_len_hp, # 输入序列长度 (来自HPO采样)
        processed_news_features_path=PROCESSED_NEWS_GLOBAL_CACHE_PATH, # 处理后的新闻特征缓存路径
        force_recompute_news=FORCE_RECOMPUTE_NEWS_GLOBAL, # 是否强制重新计算新闻特征
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET, # 是否启用时间编码
        time_freq=TIME_FREQ_IN_DATASET # 时间频率 (用于时间编码)
    ) # 创建CryptoDataset实例
    
    if len(dataset) == 0:
        print("错误: 数据集为空。") # 检查数据集是否为空
        exit()

    total_size = len(dataset) # 数据集总大小
    test_size = int(TEST_SPLIT_RATIO * total_size) # 测试集大小
    val_size = int(VALIDATION_SPLIT_RATIO * total_size) # 验证集大小
    train_size = total_size - test_size - val_size # 训练集大小
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        print(f"错误: 数据集太小无法划分. Train: {train_size}, Val: {val_size}, Test: {test_size}") # 检查数据集划分是否有效
        exit()
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) # 划分数据集
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # 训练集DataLoader
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 验证集DataLoader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # 测试集DataLoader
    edge_index = generate_edge_index(price_df_final_cols, THRESHOLD).to(DEVICE) # 生成并移动edge_index到设备

    # --- 5. Initialize Model, Loss, Optimizer ---
    print("\n--- 5. Initializing Model, Loss, Optimizer ---") # 打印初始化信息
    
    timexer_model_configs = TimeXerConfigs(
        num_nodes=dataset.num_coins, # 节点数
        price_seq_len=price_seq_len_hp, # 价格序列长度 (来自HPO采样)
        num_time_features=dataset.num_actual_time_features, # 时间特征数量
        d_model=d_model_hp, # TimeXer特征维度 (来自HPO采样)
        patch_len=patch_len_hp, # Patch长度 (来自HPO采样)
        stride=stride_hp, # Patch步长 (来自HPO采样)
        e_layers=e_layers_hp, # TimeXer编码器层数 (来自HPO采样)
        n_heads=n_heads_hp, # Attention头数量 (来自HPO采样)
        d_ff=d_ff_hp, # 前馈网络维度 (来自HPO采样)
        dropout=dropout_hp, # Dropout比例 (来自HPO采样)
        freq=TIME_FREQ_IN_DATASET # 时间频率
    ) # 创建TimeXerConfigs实例
    print(f"TimeXer Configs: enc_in={timexer_model_configs.enc_in}, seq_len={timexer_model_configs.seq_len}, d_model={timexer_model_configs.d_model}, num_time_feat={timexer_model_configs.num_time_features}") # 打印TimeXer配置

    model = TimexerGCN(
        configs=timexer_model_configs, # TimeXer配置
        hidden_dim=GCN_HIDDEN_DIM, # GCN隐藏层维度
        output_dim=GCN_OUTPUT_DIM, # GCN输出维度
        news_feature_dim=dataset.news_feature_dim, # 新闻特征原始维度
        news_processed_dim=NEWS_PROCESSED_DIM, # 新闻特征处理后维度
    ).to(DEVICE) # 创建TimexerGCN模型实例并移动到设备
    print(model) # 打印模型结构

    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # Adam优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience_hp, factor=0.5) # 学习率调度器

    # --- 6. Training Loop ---
    print("\n--- 6. Starting Training ---") # 打印训练开始信息
    best_val_loss_for_trial = float('inf') # 记录当前trial的最佳验证损失
    best_val_accuracy_for_trial = 0.0 # 记录当前trial的最佳验证准确率
    epochs_no_improve_trial = 0 # 记录验证损失连续未改善的epoch数
    
    for epoch in range(EPOCHS_PER_TRIAL): # 遍历每个trial的训练epoch
        model.train() # 设置模型为训练模式
        epoch_train_loss = 0.0 # 当前epoch训练损失总和
        epoch_train_correct = 0 # 当前epoch训练正确预测数
        epoch_train_samples = 0 # 当前epoch训练总样本数
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_PER_TRIAL} Training", leave=False) # 训练进度条
        for batch_data in train_pbar: # 遍历训练集批次
            price_seq = batch_data['price_seq'].to(DEVICE) # 价格序列数据
            x_mark_enc = batch_data['price_seq_mark'].to(DEVICE) # 时间标记数据
            news_features = batch_data['news_features'].to(DEVICE) # 新闻特征数据
            target_prices = batch_data['target_price'].to(DEVICE) # 目标价格

            target_labels = (target_prices > 0).long() # 将目标价格转换为标签 (上涨/下跌)
            
            optimizer.zero_grad() # 梯度清零
            outputs = model(price_seq, x_mark_enc, edge_index, news_features) # 模型前向传播
            
            loss = criterion(outputs.view(-1, model.mlp[-1].out_features), target_labels.view(-1)) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新模型参数
            
            epoch_train_loss += loss.item() * price_seq.size(0) # 累加训练损失
            
            preds = torch.argmax(outputs, dim=-1) # 获取预测结果
            epoch_train_correct += (preds == target_labels).sum().item() # 累加正确预测数
            epoch_train_samples += target_labels.numel() # 累加总样本数
            
            train_pbar.set_postfix(loss=loss.item()) # 更新进度条显示

        avg_train_loss = epoch_train_loss / len(train_dataset) # 计算平均训练损失
        train_accuracy = epoch_train_correct / epoch_train_samples if epoch_train_samples > 0 else 0 # 计算训练准确率
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, edge_index, DEVICE) # 在验证集上评估模型
        
        print(f"Epoch {epoch+1}/{EPOCHS_PER_TRIAL} - Train Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}") # 打印epoch结果
        
        scheduler.step(val_loss) # 根据验证损失调整学习率
        
        if val_loss < best_val_loss_for_trial: # 如果当前验证损失是当前trial最佳
            best_val_loss_for_trial = val_loss # 更新最佳验证损失
            best_val_accuracy_for_trial = val_accuracy # 更新最佳验证准确率
            torch.save(model.state_dict(), BEST_MODEL_PATH_TEMPLATE.format(trial.number)) # 保存最佳模型参数
            print(f"🚀 New best model saved to {BEST_MODEL_PATH_TEMPLATE.format(trial.number)} (Val Loss: {best_val_loss_for_trial:.4f})") # 打印模型保存信息
            epochs_no_improve_trial = 0 # 重置未改善epoch计数
        else:
            epochs_no_improve_trial += 1 # 增加未改善epoch计数
            if epochs_no_improve_trial >= scheduler_patience_hp: # 如果达到耐心值
                print(f"⏳ Early stopping after {scheduler_patience_hp} epochs with no improvement.") # 打印早停信息
                break # 早停
        
        if trial.should_prune(): # 如果Optuna决定剪枝当前trial
            raise optuna.exceptions.TrialPruned(f"Trial {trial.number} pruned at epoch {epoch+1}.") # 抛出TrialPruned异常
            
    # 保存trial的最佳验证损失、准确率和参数
    metrics = {
        "trial_number": trial.number, # trial编号
        "best_val_loss": best_val_loss_for_trial, # 最佳验证损失
        "best_val_accuracy": best_val_accuracy_for_trial, # 最佳验证准确率
        "params": dict(trial.params) # trial参数字典
    }
    metrics_path = BEST_MODEL_PATH_TEMPLATE.format(trial.number).replace(".pt", "_metrics.json") # 指标保存文件路径
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False) # 将指标保存为JSON文件
    print(f"[Trial {trial.number}] Metrics saved to {metrics_path}") # 打印指标保存路径
    return best_val_loss_for_trial # 返回最佳验证损失作为trial的结果值

# --- Main Execution for HPO ---
if __name__ == '__main__':
    print(f"Using device: {DEVICE}") # 打印使用的设备

    # Ensure the global news cache directory exists once
    if PROCESSED_NEWS_GLOBAL_CACHE_PATH:
        global_cache_dir = os.path.dirname(PROCESSED_NEWS_GLOBAL_CACHE_PATH)
        if global_cache_dir and not os.path.exists(global_cache_dir): # Ensure dirname is not empty
             os.makedirs(global_cache_dir, exist_ok=True) # 创建全局新闻缓存目录
             print(f"Ensured global news cache directory exists: {global_cache_dir}") # 打印缓存目录创建信息
    
    study = optuna.create_study(direction='minimize', # 优化方向：最小化 (验证损失)
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=N_WARMUP_STEPS_PRUNING, interval_steps=1)) # 使用MedianPruner进行剪枝
    
    try:
        study.optimize(objective, n_trials=20) # 运行Optuna优化，指定objective函数和trial数量
    except KeyboardInterrupt:
        print("HPO interrupted. Proceeding with the best trial found so far.") # 捕获中断信号

    print("Number of finished trials: ", len(study.trials)) # 打印完成的trial数量
    print("Best trial:") # 打印最佳trial信息
    trial = study.best_trial # 获取最佳trial
    print("  Value: ", trial.value) # 打印最佳trial的值 (最佳验证损失)
    print("  Params: ") # 打印最佳trial的参数
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value)) # 打印每个参数及其值