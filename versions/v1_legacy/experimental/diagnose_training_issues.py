#!/usr/bin/env python3
"""
诊断训练脚本中断问题
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import psutil

def check_system_resources():
    """检查系统资源"""
    print("=== 系统资源检查 ===")
    
    # CPU和内存
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU使用率: {cpu_percent}%")
    print(f"内存使用: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    
    # GPU信息
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            memory_total = props.total_memory / 1024**3
            print(f"GPU {i}: {props.name}")
            print(f"  总内存: {memory_total:.1f}GB")
            print(f"  已分配: {memory_allocated:.1f}GB")
            print(f"  已保留: {memory_reserved:.1f}GB")
    else:
        print("CUDA可用: 否")

def check_data_files():
    """检查数据文件"""
    print("\n=== 数据文件检查 ===")
    
    # 价格数据
    price_path = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
    if os.path.exists(price_path):
        try:
            df = pd.read_csv(price_path, index_col=0, parse_dates=True)
            print(f"✅ 价格数据: {df.shape}, 时间范围: {df.index[0]} 到 {df.index[-1]}")
            
            # 检查数据质量
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                print(f"⚠️ 发现缺失值: {null_counts.to_dict()}")
            else:
                print("✅ 无缺失值")
                
            # 检查数据范围
            print(f"数据范围: 最小值={df.min().min():.2f}, 最大值={df.max().max():.2f}")
            
        except Exception as e:
            print(f"❌ 价格数据读取错误: {e}")
    else:
        print(f"❌ 价格数据文件不存在: {price_path}")
    
    # 新闻数据
    news_path = 'scripts/analysis/crypto_new_analyzer/features'
    if os.path.exists(news_path):
        files = os.listdir(news_path)
        print(f"✅ 新闻数据: {len(files)} 个文件")
        print(f"文件列表: {files}")
    else:
        print(f"❌ 新闻数据文件夹不存在: {news_path}")

def test_model_creation():
    """测试模型创建"""
    print("\n=== 模型创建测试 ===")
    
    try:
        # 添加项目路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '.')
        sys.path.append(project_root)
        
        from models.MixModel.unified_cnn_gnn import UnifiedCnnGnn
        
        # 测试参数
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model = UnifiedCnnGnn(
            price_seq_len=60,
            num_nodes=8,
            task_type='regression',
            use_gcn=True,
            news_feature_dim=768,  # 假设的新闻特征维度
            news_processed_dim=32,
            cnn_output_channels=64,
            gcn_hidden_dim=256,
            gcn_output_dim=128,
            final_mlp_hidden_dim=256,
            num_classes=2
        ).to(device)
        
        print(f"✅ 模型创建成功")
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数: {total_params:,}")
        print(f"可训练参数数: {trainable_params:,}")
        
        # 测试前向传播
        batch_size = 4  # 小批次测试
        price_seq = torch.randn(batch_size, 60, 8).to(device)
        news_features = torch.randn(batch_size, 8, 768).to(device)
        
        # 创建边索引
        edge_index = torch.tensor([[i, j] for i in range(8) for j in range(8) if i != j]).t().to(device)
        
        with torch.no_grad():
            output = model(price_seq, edge_index, news_features=news_features)
            print(f"✅ 前向传播成功，输出形状: {output.shape}")
            
        # 检查GPU内存使用
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(device) / 1024**3
            print(f"模型GPU内存使用: {memory_used:.2f}GB")
            
    except Exception as e:
        print(f"❌ 模型创建/测试失败: {e}")
        traceback.print_exc()

def test_data_loading():
    """测试数据加载"""
    print("\n=== 数据加载测试 ===")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '.')
        sys.path.append(project_root)
        
        from scripts.analysis.crypto_new_analyzer.unified_dataset import UnifiedCryptoDataset, load_news_data
        
        # 加载价格数据
        price_df_raw = pd.read_csv('scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv', 
                                   index_col=0, parse_dates=True)
        coin_names = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
        rename_map = {f"{coin}-USDT": coin for coin in coin_names}
        price_df = price_df_raw.rename(columns=rename_map)[coin_names]
        
        print(f"✅ 价格数据加载成功: {price_df.shape}")
        
        # 加载新闻数据
        news_data_dict = load_news_data('scripts/analysis/crypto_new_analyzer/features', coin_names)
        print(f"✅ 新闻数据加载成功: {len(news_data_dict)} 个币种")
        
        # 创建数据集
        dataset = UnifiedCryptoDataset(
            price_data_df=price_df,
            news_data_dict=news_data_dict,
            seq_len=60,
            processed_news_features_path="experiments/cache/processed_news_features.pkl",
            force_recompute_news=False,
        )
        
        print(f"✅ 数据集创建成功: {len(dataset)} 个样本")
        print(f"新闻特征维度: {dataset.news_feature_dim}")
        
        # 测试获取一个样本
        sample = dataset[0]
        print(f"✅ 样本获取成功:")
        print(f"  price_seq: {sample['price_seq'].shape}")
        print(f"  target_price: {sample['target_price'].shape}")
        if 'news_features' in sample:
            print(f"  news_features: {sample['news_features'].shape}")
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        traceback.print_exc()

def suggest_solutions():
    """建议解决方案"""
    print("\n=== 建议解决方案 ===")
    
    print("1. 🔧 减少内存使用:")
    print("   - 减小批次大小: BATCH_SIZE = 16 或 8")
    print("   - 减少序列长度: PRICE_SEQ_LEN = 30")
    print("   - 减少模型参数: 降低hidden_dim")
    
    print("\n2. 🚀 优化训练:")
    print("   - 使用梯度累积")
    print("   - 启用混合精度训练")
    print("   - 清理GPU缓存")
    
    print("\n3. 🐛 调试模式:")
    print("   - 添加try-catch包装训练循环")
    print("   - 监控内存使用")
    print("   - 保存检查点")
    
    print("\n4. 📊 数据问题:")
    print("   - 检查数据预处理")
    print("   - 验证数据范围")
    print("   - 处理异常值")

def main():
    """主函数"""
    print("🔍 开始诊断训练问题...")
    print(f"时间: {datetime.now()}")
    
    check_system_resources()
    check_data_files()
    test_model_creation()
    test_data_loading()
    suggest_solutions()
    
    print(f"\n✅ 诊断完成!")

if __name__ == "__main__":
    main()
