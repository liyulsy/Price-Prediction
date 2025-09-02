#!/usr/bin/env python3
"""
使用最佳参数的示例脚本

这个脚本展示了如何加载和使用贝叶斯优化得到的最佳参数来训练WPMixer模型。

使用方法:
    python scripts/training/use_best_params_example.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from models.MixModel.unified_wpmixer import UnifiedWPMixer
from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset
from scripts.training.load_best_params import (
    load_best_params_json, create_wpmixer_config_from_params, 
    create_training_config_from_params, find_latest_params_file
)

def load_best_parameters():
    """加载最佳参数"""
    print("📂 加载最佳参数...")
    
    # 查找最新的参数文件
    params_file = find_latest_params_file()
    
    if params_file is None:
        print("❌ 未找到参数文件，使用默认参数")
        return None
    
    print(f"✅ 找到参数文件: {params_file}")
    
    # 加载参数
    params = load_best_params_json(params_file)
    
    if params is None:
        print("❌ 参数加载失败")
        return None
    
    print("✅ 参数加载成功")
    return params

def create_model_with_best_params(params):
    """使用最佳参数创建模型"""
    print("🏗️ 创建WPMixer模型...")
    
    # 创建配置对象
    configs = create_wpmixer_config_from_params(params)
    
    # 创建模型
    model = UnifiedWPMixer(
        configs=configs,
        use_gcn=False,  # 不使用GCN
        gcn_config='improved_light',
        news_feature_dim=None,  # 不使用新闻特征
        gcn_hidden_dim=256,
        gcn_output_dim=128,
        news_processed_dim=64,
        mlp_hidden_dim_1=params.get('mlp_hidden_dim_1', 1024),
        mlp_hidden_dim_2=params.get('mlp_hidden_dim_2', 512),
        num_classes=1
    )
    
    print(f"✅ 模型创建成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, configs

def prepare_data_with_best_params(params):
    """使用最佳参数准备数据"""
    print("📊 准备数据...")
    
    # 数据路径
    price_csv_path = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
    
    if not os.path.exists(price_csv_path):
        print(f"❌ 数据文件不存在: {price_csv_path}")
        return None, None, None
    
    # 加载价格数据
    price_df_raw = pd.read_csv(price_csv_path, index_col=0, parse_dates=True)
    coin_names = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
    rename_map = {f"{coin}-USDT": coin for coin in coin_names}
    price_df_full = price_df_raw.rename(columns=rename_map)[coin_names]
    
    # 确保时间索引是升序排列
    if not price_df_full.index.is_monotonic_increasing:
        price_df_full = price_df_full.sort_index()
    
    # 创建数据集
    seq_len = params.get('price_seq_len', 60)
    dataset = UnifiedCryptoDataset(
        price_data_df=price_df_full,
        news_data_dict=None,
        seq_len=seq_len,
        processed_news_features_path=None,
        force_recompute_news=False,
        time_encoding_enabled=True,
        time_freq='h',
    )
    
    # 数据集划分
    total_size = len(dataset)
    test_size = int(0.15 * total_size)
    val_size = int(0.15 * total_size)
    train_size = total_size - test_size - val_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    # 创建数据子集
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"✅ 数据准备完成")
    print(f"   序列长度: {seq_len}")
    print(f"   训练集: {len(train_dataset)}")
    print(f"   验证集: {len(val_dataset)}")
    print(f"   测试集: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

def train_model_with_best_params(model, train_dataset, val_dataset, params):
    """使用最佳参数训练模型"""
    print("🏃 开始训练模型...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建训练配置
    training_config = create_training_config_from_params(params)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False)
    
    # 设置训练组件
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.8)
    
    print(f"📋 训练配置:")
    for key, value in training_config.items():
        if key in ['learning_rate', 'weight_decay']:
            print(f"   {key}: {value:.8f}")
        else:
            print(f"   {key}: {value}")
    
    # 训练循环（简化版本，仅训练几个epoch作为示例）
    num_epochs = training_config['epochs']  # 使用最佳参数中的完整epoch数
    print(f"🔄 开始训练 {num_epochs} 个epoch（示例）...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            price_seq = batch_data['price_seq'].to(device)
            target_data = batch_data['target_price'].to(device)
            
            optimizer.zero_grad()
            outputs = model(price_data=price_seq)
            outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, target_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # 只处理前几个batch作为示例
            if batch_idx >= 10:
                break
        
        avg_train_loss = train_loss / min(len(train_loader), 11)
        
        # 验证阶段（简化）
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                price_seq = batch_data['price_seq'].to(device)
                target_data = batch_data['target_price'].to(device)
                
                outputs = model(price_data=price_seq)
                outputs = outputs.squeeze(-1)
                
                loss = criterion(outputs, target_data)
                val_loss += loss.item()
                
                # 只处理前几个batch作为示例
                if batch_idx >= 5:
                    break
        
        avg_val_loss = val_loss / min(len(val_loader), 6)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}")
    
    print("✅ 训练完成（示例）")
    return model

def main():
    """主函数"""
    print("🎯 使用最佳参数训练WPMixer示例")
    print("="*50)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载最佳参数
    params = load_best_parameters()
    if params is None:
        print("❌ 无法加载参数，退出")
        return
    
    # 2. 创建模型
    model, configs = create_model_with_best_params(params)
    
    # 3. 准备数据
    train_dataset, val_dataset, test_dataset = prepare_data_with_best_params(params)
    if train_dataset is None:
        print("❌ 数据准备失败，退出")
        return
    
    # 4. 训练模型
    trained_model = train_model_with_best_params(model, train_dataset, val_dataset, params)
    
    print(f"\n🎉 示例完成!")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n💡 这只是一个使用最佳参数的示例。")
    print(f"   实际使用时，请根据需要调整训练轮数和其他设置。")
    print(f"\n📁 最佳参数文件位置:")
    params_file = find_latest_params_file()
    if params_file:
        print(f"   {params_file}")

if __name__ == '__main__':
    main()
