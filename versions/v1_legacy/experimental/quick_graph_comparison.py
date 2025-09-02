#!/usr/bin/env python3
"""
快速比较最有前景的图构建方法
基于图分析结果，重点测试domain_knowledge, dynamic, 和original方法
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
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
CACHE_DIR = "experiments/cache/quick_graph_comparison"

# 训练参数（快速测试）
BATCH_SIZE = 64
EPOCHS = 3  # 快速测试
LEARNING_RATE = 0.001
PRICE_SEQ_LEN = 60
VALIDATION_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# 基于分析结果选择的最有前景的方法
SELECTED_METHODS = {
    'original': {'threshold': 0.6},
    'domain_knowledge': {'coin_names': COIN_NAMES},
    'dynamic': {'window_size': 168, 'overlap': 24},
}

class TimeMixerBaseConfigs:
    """简化的TimeMixer配置"""
    def __init__(self, num_nodes, price_seq_len, num_time_features):
        self.enc_in = num_nodes
        self.seq_len = price_seq_len
        self.d_model = 32  # 减小模型以加快训练
        self.num_time_features = num_time_features
        self.task_type = 'regression'
        self.down_sampling_layers = 1  # 减少层数
        self.down_sampling_window = 2
        self.down_sampling_method = 'avg'
        self.e_layers = 1  # 减少层数
        self.n_heads = 2  # 减少注意力头
        self.d_ff = 64
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
    
    # 使用最近的数据以加快训练
    price_df_full = price_df_full.tail(5000)  # 只使用最近5000个数据点
    
    # 标准化
    scaler = StandardScaler()
    price_df_values = scaler.fit_transform(price_df_full)
    price_df = pd.DataFrame(price_df_values, columns=price_df_full.columns, index=price_df_full.index)
    
    # 创建数据集
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df,
        news_data_dict=None,  # 不使用新闻数据
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
        use_gcn=True,
        news_feature_dim=None,
        gcn_hidden_dim=64,  # 减小GCN维度
        gcn_output_dim=32,
        news_processed_dim=16,
        prediction_head_dim=32,
        mlp_hidden_dim=128,
        num_classes=1
    ).to(DEVICE)
    
    return model, configs

def quick_train_and_evaluate(model, train_loader, val_loader, test_loader, edge_index, edge_weights, method_name):
    """快速训练和评估"""
    print(f"Training {method_name}...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    # 快速训练
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
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
            batch_count += 1
            
            # 限制每个epoch的batch数量以加快训练
            if batch_count >= 20:
                break
        
        avg_train_loss = train_loss / batch_count
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
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
                val_batch_count += 1
                
                if val_batch_count >= 10:  # 限制验证batch数量
                    break
        
        avg_val_loss = val_loss / val_batch_count
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f"  Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
    
    # 测试评估
    model.eval()
    test_preds = []
    test_targets = []
    test_batch_count = 0
    
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
            
            test_batch_count += 1
            if test_batch_count >= 15:  # 限制测试batch数量
                break
    
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    # 计算指标
    mae = mean_absolute_error(test_targets, test_preds)
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_targets, test_preds)
    
    # 计算新的MAE
    total_true_sum = np.sum(test_targets)
    total_pred_sum = np.sum(test_preds)
    new_mae = total_true_sum / total_pred_sum if total_pred_sum != 0 else float('inf')
    
    return {
        'method': method_name,
        'best_val_loss': best_val_loss,
        'test_mae': mae,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2': r2,
        'new_mae': new_mae
    }

def main():
    """主函数"""
    print("=== Quick Graph Method Comparison ===")
    print(f"Device: {DEVICE}")
    print(f"Testing methods: {list(SELECTED_METHODS.keys())}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 准备数据
    price_df_full, train_loader, val_loader, test_loader, dataset, scaler = prepare_data()
    print(f"Dataset size: {len(dataset)} samples")
    
    results = []
    
    # 测试选定的图构建方法
    for method_name, method_params in SELECTED_METHODS.items():
        print(f"\n{'='*50}")
        print(f"Testing {method_name.upper()}")
        print(f"{'='*50}")
        
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
            print(f"Graph: {graph_props['num_edges']} edges, density={graph_props['density']:.4f}")
            
            # 创建模型
            model, configs = create_model(dataset)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            # 训练和评估
            result = quick_train_and_evaluate(model, train_loader, val_loader, test_loader, 
                                            edge_index, edge_weights, method_name)
            
            # 添加图属性
            result.update({
                'num_edges': graph_props['num_edges'],
                'density': graph_props['density'],
                'avg_degree': graph_props['avg_degree']
            })
            
            results.append(result)
            
            print(f"✅ Results:")
            print(f"   MAE: {result['test_mae']:.6f}")
            print(f"   New MAE: {result['new_mae']:.6f}")
            print(f"   R²: {result['test_r2']:.6f}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 分析结果
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}")
        
        # 按MAE排序
        results_sorted = sorted(results, key=lambda x: x['test_mae'])
        
        print("\nRanking by MAE (lower is better):")
        for i, result in enumerate(results_sorted, 1):
            print(f"{i}. {result['method'].upper()}")
            print(f"   MAE: {result['test_mae']:.6f}")
            print(f"   New MAE: {result['new_mae']:.6f} (closer to 1.0 is better)")
            print(f"   R²: {result['test_r2']:.6f}")
            print(f"   Graph: {result['num_edges']} edges, density={result['density']:.4f}")
        
        # 保存结果
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(CACHE_DIR, f"quick_comparison_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        best_method = results_sorted[0]
        print(f"\n🏆 BEST METHOD: {best_method['method'].upper()}")
        print(f"   MAE: {best_method['test_mae']:.6f}")
        print(f"   New MAE: {best_method['new_mae']:.6f}")
        
        print(f"\nResults saved to: {results_file}")
        
        # 推荐
        print(f"\n💡 RECOMMENDATION:")
        print(f"Use '{best_method['method']}' method for your full training.")
        print(f"Update GRAPH_METHOD in train_multiscale.py to '{best_method['method']}'")
    
    print(f"\n=== Quick Comparison Complete ===")

if __name__ == "__main__":
    main()
