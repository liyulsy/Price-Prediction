import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import json
import pickle
from datetime import datetime

# Import the unified Timexer model and dataset components
from models.MixModel.unified_timexer_gcn import UnifiedTimexerGCN
from crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index

# --- Fixed Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TASK_TYPE = 'regression'
USE_NEWS_FEATURES = False
USE_GCN = False

# --- Data Paths ---
PRICE_CSV_PATH = 'datafiles/price_data/1H.csv'
NEWS_FEATURES_FOLDER = 'crypto_new_analyzer/features'
CACHE_DIR = "cache"
OPTIMIZATION_RESULTS_DIR = "optimization_results"

# --- Dataset Parameters ---
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
# PRICE_SEQ_LEN = 60  # 移除固定值，将在优化中搜索
THRESHOLD = 0.6
NORM_TYPE = 'minmax'
TIME_ENCODING_ENABLED_IN_DATASET = True
TIME_FREQ_IN_DATASET = 'h'

# --- Optimization Parameters ---
N_TRIALS = 50  # 优化试验次数
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 30  # 减少训练轮数以加快优化速度
EARLY_STOPPING_PATIENCE = 8

# --- TimeXer-Specific Configurations ---
class TimeXerConfigs:
    """A configuration class for the TimeXer model's specific needs."""
    def __init__(self, num_coins, price_seq_len, pred_len=1, d_model=64, freq='h',
                embed_type='timeF', patch_len=16, stride=8, e_layers=1,
                d_ff=256, dropout=0.1, fc_dropout=0.1, head_dropout=0.1,
                act='gelu', n_heads=4, factor=5, use_norm=False, task_type='regression'):
        self.enc_in = num_coins
        self.dec_in = num_coins
        self.c_out = num_coins
        self.seq_len = price_seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.freq = freq
        self.embed = embed_type
        self.patch_len = patch_len
        self.stride = stride
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.factor = factor
        self.activation = act
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.task_type = task_type

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(model, data_loader, criterion, edge_index, device, task_type, scaler=None):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating", leave=False):
            price_seq = batch_data['price_seq'].to(device)
            price_seq_mark = batch_data['price_seq_mark'].to(device)
            target_data = batch_data['target_price'].to(device)
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(device)

            x_enc = price_seq
            outputs = model(x_enc, price_seq_mark, edge_index=edge_index, news_features=news_features)

            if task_type == 'classification':
                targets = (target_data > 0).long()
                loss = criterion(outputs.view(-1, 2), targets.view(-1))
                preds = torch.argmax(outputs, dim=-1)
            else: # regression
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
    if task_type == 'regression':
        if scaler:
            num_coins = all_preds.shape[1]
            original_preds = scaler.inverse_transform(all_preds.reshape(-1, num_coins))
            original_targets = scaler.inverse_transform(all_targets.reshape(-1, num_coins))
        else:
            original_preds = all_preds
            original_targets = all_targets
            
        metrics['mae'] = mean_absolute_error(original_targets, original_preds)
        metrics['mse'] = mean_squared_error(original_targets, original_preds)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(original_targets, original_preds)
        metrics['mape'] = mean_absolute_percentage_error(original_targets, original_preds)

    return metrics

def train_and_evaluate(trial, price_seq_len, train_loader, val_loader, edge_index, scaler, dataset):
    """训练模型并评估性能，用于贝叶斯优化。优化目标是最大化 R2 分数。"""
    
    # 超参数搜索空间
    # price_seq_len 从 objective 函数传入
    
    # 静态定义 patch_len 搜索空间，然后检查约束并剪枝
    patch_len = trial.suggest_categorical('patch_len', [8, 16, 24, 32])
    if patch_len > price_seq_len:
        raise optuna.exceptions.TrialPruned(f"Pruned because patch_len ({patch_len}) > price_seq_len ({price_seq_len}).")

    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    stride = trial.suggest_categorical('stride', [2, 4, 8])
    e_layers = trial.suggest_int('e_layers', 2, 6)
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    d_ff = trial.suggest_categorical('d_ff', [256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.05, 0.3, step=0.05)
    
    # MLP 参数
    mlp_hidden_dim_1 = trial.suggest_categorical('mlp_hidden_dim_1', [256, 512, 1024])
    mlp_hidden_dim_2 = trial.suggest_categorical('mlp_hidden_dim_2', [128, 256, 512])
    
    # 训练参数
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # 创建配置
    timexer_configs = TimeXerConfigs(
        num_coins=len(COIN_NAMES), 
        price_seq_len=price_seq_len,  # 使用优化的序列长度
        d_model=d_model,
        patch_len=patch_len,
        stride=stride,
        e_layers=e_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        task_type=TASK_TYPE
    )
    
    # 创建模型
    model = UnifiedTimexerGCN(
        configs=timexer_configs,
        gcn_hidden_dim=256,
        gcn_output_dim=128,
        use_gcn=USE_GCN,
        news_feature_dim=dataset.news_feature_dim if USE_NEWS_FEATURES else None,
        news_processed_dim=64,
        mlp_hidden_dim_1=mlp_hidden_dim_1,
        mlp_hidden_dim_2=mlp_hidden_dim_2,
        num_classes=2
    ).to(DEVICE)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7, min_lr=1e-6)
    
    # 训练循环
    best_val_r2 = -float('inf')
    best_val_metrics = {}
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_data in train_loader:
            price_seq = batch_data['price_seq'].to(DEVICE)
            price_seq_mark = batch_data['price_seq_mark'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)
            news_features = batch_data.get('news_features')
            if news_features is not None:
                news_features = news_features.to(DEVICE)

            optimizer.zero_grad()
            
            x_enc = price_seq
            outputs = model(x_enc, price_seq_mark, edge_index=edge_index, news_features=news_features)
            
            targets = target_data
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * price_seq.size(0)

        # 验证
        val_metrics = evaluate_model(model, val_loader, criterion, edge_index, DEVICE, TASK_TYPE, scaler)
        val_loss = val_metrics['loss']
        val_r2 = val_metrics.get('r2', -float('inf'))
        scheduler.step(val_loss) # 调度器仍然基于损失
        
        # 早停基于 R2 分数
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break
    
    # 记录其他指标以供分析
    if best_val_metrics:
        trial.set_user_attr("best_epoch_val_loss", best_val_metrics.get('loss'))
        trial.set_user_attr("best_epoch_val_mape", best_val_metrics.get('mape'))
        trial.set_user_attr("best_epoch_val_r2", best_val_metrics.get('r2'))

    # 返回 R2 作为优化目标
    return best_val_r2

def objective(trial):
    """Optuna 优化目标函数"""
    
    # 首先定义需要影响数据集的超参数
    price_seq_len = trial.suggest_categorical('price_seq_len', [30, 60, 90, 120, 180])
    
    # 数据加载和预处理
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]

    edge_index = generate_edge_index(price_df_full, THRESHOLD).to(DEVICE) if USE_GCN else None

    fit_train_size = int(len(price_df_full) * (1 - VALIDATION_SPLIT_RATIO - TEST_SPLIT_RATIO))
    price_df_for_scaler = price_df_full.iloc[:fit_train_size]
    
    if NORM_TYPE == 'standard': 
        scaler = StandardScaler()
    elif NORM_TYPE == 'minmax': 
        scaler = MinMaxScaler()
    else: 
        scaler = None

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
        seq_len=price_seq_len,  # 使用动态序列长度
        processed_news_features_path=None,
        force_recompute_news=False,
        time_encoding_enabled=TIME_ENCODING_ENABLED_IN_DATASET,
        time_freq=TIME_FREQ_IN_DATASET
    )
    
    total_size = len(dataset)
    test_size = int(TEST_SPLIT_RATIO * total_size)
    val_size = int(VALIDATION_SPLIT_RATIO * total_size)
    train_size = total_size - test_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 训练和评估，返回 R2
    val_r2 = train_and_evaluate(trial, price_seq_len, train_loader, val_loader, edge_index, scaler, dataset)
    
    return val_r2

def save_optimization_results(study, best_params, best_value):
    """保存优化结果"""
    os.makedirs(OPTIMIZATION_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存最佳参数
    results = {
        'timestamp': timestamp,
        'best_params': best_params,
        'best_value_R2': best_value,
        'n_trials': len(study.trials),
        'optimization_history': [
            {
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name,
                'user_attrs': trial.user_attrs
            }
            for trial in study.trials
        ]
    }
    
    # 保存为 JSON
    json_path = os.path.join(OPTIMIZATION_RESULTS_DIR, f'optimization_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存 Optuna study 对象
    study_path = os.path.join(OPTIMIZATION_RESULTS_DIR, f'study_{timestamp}.pkl')
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    print(f"优化结果已保存到:")
    print(f"  JSON: {json_path}")
    print(f"  Study: {study_path}")
    
    return json_path, study_path

if __name__ == '__main__':
    print(f"=== 贝叶斯优化开始 ===")
    print(f"设备: {DEVICE}")
    print(f"试验次数: {N_TRIALS}")
    print(f"任务类型: {TASK_TYPE}")
    print(f"使用新闻: {USE_NEWS_FEATURES}")
    print(f"使用GCN: {USE_GCN}")
    print(f"归一化: {NORM_TYPE}")
    print(f"序列长度: [30, 60, 90, 120, 180] (优化中)")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"早停耐心: {EARLY_STOPPING_PATIENCE}")
    print(f"====================\n")
    
    # 创建 Optuna study
    study = optuna.create_study(
        direction='maximize',  # 目标：最大化 R2
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # 开始优化
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # 获取最佳结果
    best_params = study.best_params
    best_value = study.best_value
    
    print(f"\n=== 优化完成 ===")
    print(f"最佳验证 R²: {best_value:.6f}")
    print(f"最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 保存结果
    json_path, study_path = save_optimization_results(study, best_params, best_value)
    
    # 生成可直接使用的配置代码
    config_code = f"""
# 贝叶斯优化得到的最佳参数配置
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# 最佳验证 R²: {best_value:.6f}

# 数据集参数
PRICE_SEQ_LEN = {best_params['price_seq_len']}

# 模型参数
D_MODEL = {best_params['d_model']}
PATCH_LEN = {best_params['patch_len']}
STRIDE = {best_params['stride']}
E_LAYERS = {best_params['e_layers']}
N_HEADS = {best_params['n_heads']}
D_FF = {best_params['d_ff']}
DROPOUT_TIMEXER = {best_params['dropout']}

# MLP 参数
MLP_HIDDEN_DIM_1 = {best_params['mlp_hidden_dim_1']}
MLP_HIDDEN_DIM_2 = {best_params['mlp_hidden_dim_2']}

# 训练参数
LEARNING_RATE = {best_params['learning_rate']}
WEIGHT_DECAY = {best_params['weight_decay']}
"""
    
    config_path = os.path.join(OPTIMIZATION_RESULTS_DIR, f'best_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py')
    with open(config_path, 'w') as f:
        f.write(config_code)
    
    print(f"\n最佳配置代码已保存到: {config_path}")
    print(f"\n=== 优化完成 ===") 