#!/usr/bin/env python3
"""
比较不同图构建方法在预测任务上的效果
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from models.MixModel.unified_multiscale_timemixer import UnifiedMultiScaleTimeMixer
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
from dataloader.gnn_loader import generate_edge_index, generate_advanced_edge_index, analyze_graph_properties

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
NEWS_FEATURES_FOLDER = 'scripts/analysis/crypto_new_analyzer/features'
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
CACHE_DIR = "experiments/cache/graph_comparison"

# 训练参数
BATCH_SIZE = 32
EPOCHS = 5  # 快速测试
LEARNING_RATE = 0.001
PRICE_SEQ_LEN = 60
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# 图构建方法配置
GRAPH_METHODS = {
    'original': {'threshold': 0.6},
    'multi_layer': {'correlation_threshold': 0.3, 'volatility_threshold': 0.5, 'trend_threshold': 0.4},
    'dynamic': {'window_size': 168, 'overlap': 24},
    'domain_knowledge': {'coin_names': COIN_NAMES},
    'attention_based': {'top_k': 3, 'use_returns': True}
}

class TimeMixerBaseConfigs:
    """TimeMixer配置类"""
    def __init__(self, num_nodes, price_seq_len, num_time_features):
        self.enc_in = num_nodes
        self.seq_len = price_seq_len
        self.d_model = 64
        self.num_time_features = num_time_features
        self.task_type = 'regression'
        self.down_sampling_layers = 2
        self.down_sampling_window = 2
        self.down_sampling_method = 'avg'
        self.e_layers = 2
        self.n_heads = 4
        self.d_ff = 128
        self.dropout = 0.1
        self.channel_independence = True
        self.pred_len = 1
        self.freq = 'h'
        self.embed = 'timeF'
        self.use_norm = False
        self.decomp_method = 'moving_avg'
        self.moving_avg = 25
        self.dec_in = num_nodes
        self.c_out = num_nodes
        self.label_len = int(price_seq_len * 0.5)
        self.output_attention = False
        self.factor = 5
        self.activation = 'gelu'

def prepare_data():
    """准备数据"""
    print("Preparing data...")
    
    # 加载价格数据
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]
    
    # 标准化
    scaler = StandardScaler()
    price_df_values = scaler.fit_transform(price_df_full)
    price_df = pd.DataFrame(price_df_values, columns=price_df_full.columns, index=price_df_full.index)
    
    # 创建数据集
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        news_data_dict=None,  # 不使用新闻数据以简化测试
        seq_len=PRICE_SEQ_LEN,
        processed_news_features_path=None,
        force_recompute_news=False,
        time_encoding_enabled=True,
        time_freq='h',
    )
    
    # 分割数据集
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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return price_df_full, train_loader, val_loader, test_loader, dataset, scaler

def create_model(dataset):
    """创建模型"""
    configs = TimeMixerBaseConfigs(
        num_nodes=dataset.num_coins,
        price_seq_len=PRICE_SEQ_LEN,
        num_time_features=dataset.num_actual_time_features
    )
    
    model = UnifiedMultiScaleTimeMixer(
        configs=configs,
        use_gcn=True,  # 启用GCN
        news_feature_dim=None,  # 不使用新闻
        gcn_hidden_dim=128,
        gcn_output_dim=64,
        news_processed_dim=32,
        prediction_head_dim=64,
        mlp_hidden_dim=256,
        num_classes=1
    ).to(DEVICE)
    
    return model, configs

def train_and_evaluate(model, train_loader, val_loader, test_loader, edge_index, edge_weights, method_name):
    """训练和评估模型"""
    print(f"\nTraining with {method_name} graph method...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_data in train_loader:
            price_seq = batch_data['price_seq'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)
            x_mark_enc = batch_data.get('price_seq_mark')
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(price_seq, x_mark_enc, edge_index=edge_index, edge_weight=edge_weights)
            loss = criterion(outputs, target_data)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                price_seq = batch_data['price_seq'].to(DEVICE)
                target_data = batch_data['target_price'].to(DEVICE)
                x_mark_enc = batch_data.get('price_seq_mark')
                if x_mark_enc is not None:
                    x_mark_enc = x_mark_enc.to(DEVICE)
                
                outputs = model(price_seq, x_mark_enc, edge_index=edge_index, edge_weight=edge_weights)
                loss = criterion(outputs, target_data)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"  Epoch {epoch+1}/{EPOCHS}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    # 测试评估
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            price_seq = batch_data['price_seq'].to(DEVICE)
            target_data = batch_data['target_price'].to(DEVICE)
            x_mark_enc = batch_data.get('price_seq_mark')
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc.to(DEVICE)
            
            outputs = model(price_seq, x_mark_enc, edge_index=edge_index, edge_weight=edge_weights)
            test_preds.append(outputs.cpu().numpy())
            test_targets.append(target_data.cpu().numpy())
    
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # 计算指标
    mae = mean_absolute_error(test_targets, test_preds)
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets, test_preds)
    
    return {
        'method': method_name,
        'best_val_loss': best_val_loss,
        'test_mae': mae,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2': r2
    }

def main():
    """主函数"""
    print("=== Graph Method Comparison for Crypto Prediction ===")
    print(f"Device: {DEVICE}")
    print(f"Coins: {COIN_NAMES}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 准备数据
    price_df_full, train_loader, val_loader, test_loader, dataset, scaler = prepare_data()
    
    results = []
    
    # 测试每种图构建方法
    for method_name, method_params in GRAPH_METHODS.items():
        print(f"\n{'='*60}")
        print(f"Testing {method_name.upper()} method")
        print(f"{'='*60}")
        
        try:
            # 构建图
            if method_name == 'original':
                edge_index = generate_edge_index(price_df_full, **method_params).to(DEVICE)
                edge_weights = None
            else:
                edge_index, edge_weights = generate_advanced_edge_index(
                    price_df_full, method=method_name, **method_params
                )
                edge_index = edge_index.to(DEVICE)
                edge_weights = edge_weights.to(DEVICE) if edge_weights is not None else None
            
            # 分析图属性
            graph_props = analyze_graph_properties(edge_index, edge_weights, len(COIN_NAMES))
            print(f"Graph properties: {graph_props['num_edges']} edges, density={graph_props['density']:.4f}")
            
            # 创建和训练模型
            model, configs = create_model(dataset)
            
            # 训练和评估
            result = train_and_evaluate(model, train_loader, val_loader, test_loader, 
                                      edge_index, edge_weights, method_name)
            
            # 添加图属性到结果
            result.update({
                'num_edges': graph_props['num_edges'],
                'density': graph_props['density'],
                'avg_degree': graph_props['avg_degree']
            })
            
            results.append(result)
            
            print(f"✅ {method_name} completed successfully!")
            print(f"   Test MAE: {result['test_mae']:.6f}")
            print(f"   Test R²: {result['test_r2']:.6f}")
            
        except Exception as e:
            print(f"❌ Error with {method_name}: {str(e)}")
            continue
    
    # 保存和显示结果
    if results:
        results_df = pd.DataFrame(results)
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(CACHE_DIR, f"graph_method_comparison_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n{'='*80}")
        print("FINAL RESULTS COMPARISON")
        print(f"{'='*80}")
        
        # 按测试MAE排序
        results_df_sorted = results_df.sort_values('test_mae')
        
        print("\nRanking by Test MAE (lower is better):")
        for i, (_, row) in enumerate(results_df_sorted.iterrows(), 1):
            print(f"{i}. {row['method'].upper()}")
            print(f"   MAE: {row['test_mae']:.6f} | R²: {row['test_r2']:.6f}")
            print(f"   Edges: {row['num_edges']} | Density: {row['density']:.4f}")
        
        best_method = results_df_sorted.iloc[0]
        print(f"\n🏆 BEST PERFORMING METHOD: {best_method['method'].upper()}")
        print(f"   Test MAE: {best_method['test_mae']:.6f}")
        print(f"   Test R²: {best_method['test_r2']:.6f}")
        
        print(f"\nDetailed results saved to: {results_file}")
    
    print(f"\n=== Comparison Complete ===")

if __name__ == "__main__":
    main()
