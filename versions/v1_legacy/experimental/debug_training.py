#!/usr/bin/env python3
"""
调试版本的训练脚本 - 用于诊断训练中断问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import traceback

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 简化的配置
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8  # 非常小的批次大小
PRICE_SEQ_LEN = 20  # 短序列
EPOCHS = 2  # 少量epoch用于测试

print(f"🔧 调试模式启动")
print(f"设备: {DEVICE}")
print(f"批次大小: {BATCH_SIZE}")
print(f"序列长度: {PRICE_SEQ_LEN}")

def check_gpu_memory():
    """检查GPU内存使用"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(DEVICE) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**3
        print(f"GPU内存 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")
        return memory_allocated
    return 0

def main():
    """主函数"""
    try:
        print("\n=== 步骤1: 检查初始GPU内存 ===")
        check_gpu_memory()
        
        print("\n=== 步骤2: 加载数据 ===")
        # 加载价格数据
        price_path = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
        if not os.path.exists(price_path):
            print(f"❌ 价格数据文件不存在: {price_path}")
            return
            
        price_df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True)
        coin_names = ['BTC', 'ETH', 'BNB', 'XRP']  # 只使用4个币种
        rename_map = {f"{coin}-USDT": coin for coin in coin_names}
        price_df_full = price_df_raw.rename(columns=rename_map)[coin_names]
        
        print(f"✅ 价格数据加载成功: {price_df_full.shape}")
        
        # 数据归一化
        scaler = StandardScaler()
        price_values = scaler.fit_transform(price_df_full)
        price_df = pd.DataFrame(price_values, columns=price_df_full.columns, index=price_df_full.index)
        
        print("\n=== 步骤3: 创建简单数据集 ===")
        # 创建简单的数据集
        sequences = []
        targets = []
        
        for i in range(PRICE_SEQ_LEN, len(price_df) - 1):
            seq = price_df.iloc[i-PRICE_SEQ_LEN:i].values
            target = price_df.iloc[i+1].values
            sequences.append(seq)
            targets.append(target)
            
            if len(sequences) >= 1000:  # 限制数据量
                break
        
        sequences = torch.FloatTensor(sequences)
        targets = torch.FloatTensor(targets)
        
        print(f"✅ 数据集创建成功: {sequences.shape}, {targets.shape}")
        
        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(sequences, targets)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        
        print("\n=== 步骤4: 创建简单模型 ===")
        check_gpu_memory()
        
        # 简单的LSTM模型
        class SimpleLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])
                return output
        
        model = SimpleLSTM(input_size=len(coin_names), hidden_size=32, output_size=len(coin_names))
        model = model.to(DEVICE)
        
        print(f"✅ 模型创建成功")
        check_gpu_memory()
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print("\n=== 步骤5: 开始训练 ===")
        
        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
            
            # 训练
            model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_idx, (batch_seq, batch_target) in enumerate(train_loader):
                try:
                    batch_seq = batch_seq.to(DEVICE)
                    batch_target = batch_target.to(DEVICE)
                    
                    optimizer.zero_grad()
                    output = model(batch_seq)
                    loss = criterion(output, batch_target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"  批次 {batch_idx}: Loss = {loss.item():.4f}")
                        check_gpu_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"❌ GPU内存不足在批次 {batch_idx}: {e}")
                        torch.cuda.empty_cache()
                        return
                    else:
                        raise e
                        
            avg_train_loss = train_loss / train_batches
            print(f"平均训练损失: {avg_train_loss:.4f}")
            
            # 验证
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    batch_seq = batch_seq.to(DEVICE)
                    batch_target = batch_target.to(DEVICE)
                    
                    output = model(batch_seq)
                    loss = criterion(output, batch_target)
                    val_loss += loss.item()
                    val_batches += 1
                    
            avg_val_loss = val_loss / val_batches
            print(f"平均验证损失: {avg_val_loss:.4f}")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            check_gpu_memory()
        
        print("\n✅ 调试训练完成!")
        
    except Exception as e:
        print(f"❌ 调试训练失败: {e}")
        traceback.print_exc()
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("\n🔧 建议的解决方案:")
        print("1. 进一步减少批次大小")
        print("2. 减少序列长度")
        print("3. 使用更简单的模型")
        print("4. 检查数据预处理")
        print("5. 确保正确的conda环境")

if __name__ == "__main__":
    main()
