#!/usr/bin/env python3
"""
分类任务诊断脚本
分析为什么四个消融实验的准确率很接近
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import sys

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# 配置参数（与train_cnn.py保持一致）
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PREDICTION_TARGET = 'diff'  # 与train_cnn.py保持一致

def analyze_classification_data():
    """分析分类数据的分布特征"""
    print("🔍 加载和分析分类数据...")
    
    # 1. 加载数据
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df_full = price_df_raw.rename(columns=rename_map)[COIN_NAMES]
    
    # 2. 计算差分
    if PREDICTION_TARGET == 'diff':
        df_target = price_df_full.diff().dropna()
        target_name = "价格差分"
    elif PREDICTION_TARGET == 'return':
        df_target = price_df_full.pct_change().dropna()
        target_name = "价格变化率"
    else:
        print("❌ 只支持diff和return分类任务")
        return
    
    print(f"📊 分析{target_name}数据...")
    print(f"数据形状: {df_target.shape}")
    print(f"时间范围: {df_target.index.min()} 到 {df_target.index.max()}")
    
    # 3. 分析每个币种的涨跌分布
    print(f"\n📈 各币种涨跌分布分析:")
    print("=" * 60)
    
    overall_stats = {}
    for coin in COIN_NAMES:
        coin_data = df_target[coin]
        
        # 计算涨跌统计
        total_samples = len(coin_data)
        up_samples = (coin_data > 0).sum()
        down_samples = (coin_data <= 0).sum()
        up_ratio = up_samples / total_samples
        
        # 计算涨跌幅度统计
        up_values = coin_data[coin_data > 0]
        down_values = coin_data[coin_data <= 0]
        
        overall_stats[coin] = {
            'total': total_samples,
            'up_count': up_samples,
            'down_count': down_samples,
            'up_ratio': up_ratio,
            'avg_up': up_values.mean() if len(up_values) > 0 else 0,
            'avg_down': down_values.mean() if len(down_values) > 0 else 0,
            'std_up': up_values.std() if len(up_values) > 0 else 0,
            'std_down': down_values.std() if len(down_values) > 0 else 0
        }
        
        print(f"{coin:>6}: 涨{up_samples:>6}({up_ratio:>6.1%}) | 跌{down_samples:>6}({1-up_ratio:>6.1%}) | "
              f"平均涨幅{up_values.mean():>8.2f} | 平均跌幅{down_values.mean():>8.2f}")
    
    # 4. 整体统计
    all_values = df_target.values.flatten()
    total_up = (all_values > 0).sum()
    total_down = (all_values <= 0).sum()
    overall_up_ratio = total_up / len(all_values)
    
    print(f"\n🌍 整体统计:")
    print(f"总样本数: {len(all_values):,}")
    print(f"涨的样本: {total_up:,} ({overall_up_ratio:.1%})")
    print(f"跌的样本: {total_down:,} ({1-overall_up_ratio:.1%})")
    
    # 5. 检查类别不平衡程度
    imbalance_ratio = min(overall_up_ratio, 1-overall_up_ratio) / max(overall_up_ratio, 1-overall_up_ratio)
    print(f"类别平衡度: {imbalance_ratio:.3f} (1.0为完全平衡)")
    
    if imbalance_ratio < 0.8:
        print("⚠️  存在类别不平衡问题！")
    else:
        print("✅ 类别分布相对平衡")
    
    # 6. 分析为什么准确率接近
    print(f"\n🤔 准确率接近的可能原因分析:")
    
    # 基线准确率（总是预测多数类）
    baseline_acc = max(overall_up_ratio, 1-overall_up_ratio)
    print(f"1. 基线准确率（总是预测多数类）: {baseline_acc:.1%}")
    
    # 随机预测准确率
    random_acc = 0.5
    print(f"2. 随机预测准确率: {random_acc:.1%}")
    
    # 如果模型准确率都在基线附近，说明模型没有学到有用信息
    if baseline_acc > 0.6:
        print("⚠️  基线准确率较高，模型可能只是在学习数据分布而非真正的预测模式")
    
    # 7. 检查时间序列的自相关性
    print(f"\n📊 时间序列自相关性分析:")
    for coin in COIN_NAMES[:3]:  # 只分析前3个币种
        coin_data = df_target[coin]
        # 计算1步自相关
        autocorr_1 = coin_data.autocorr(lag=1)
        print(f"{coin}: 1步自相关 = {autocorr_1:.4f}")
        
        if abs(autocorr_1) < 0.1:
            print(f"  → {coin}的价格变化接近随机游走，难以预测")
    
    # 8. 保存分析结果
    results_df = pd.DataFrame(overall_stats).T
    results_df.to_csv('experiments/cache/classification_analysis.csv')
    print(f"\n💾 分析结果已保存到: experiments/cache/classification_analysis.csv")
    
    return overall_stats, df_target

def plot_distribution_analysis(df_target):
    """绘制分布分析图"""
    print("\n📊 生成分布分析图...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, coin in enumerate(COIN_NAMES):
        ax = axes[i]
        coin_data = df_target[coin]
        
        # 绘制直方图
        ax.hist(coin_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零线')
        ax.set_title(f'{coin} 价格差分分布')
        ax.set_xlabel('价格差分')
        ax.set_ylabel('频次')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/cache/classification_distribution.png', dpi=300, bbox_inches='tight')
    print("📊 分布图已保存到: experiments/cache/classification_distribution.png")
    plt.close()

if __name__ == '__main__':
    print("🚀 开始分类任务诊断...")
    
    # 创建输出目录
    os.makedirs('experiments/cache', exist_ok=True)
    
    # 执行分析
    stats, df_target = analyze_classification_data()
    plot_distribution_analysis(df_target)
    
    print("\n✅ 诊断完成！")
    print("\n💡 建议:")
    print("1. 检查 experiments/cache/classification_analysis.csv 了解详细统计")
    print("2. 查看 experiments/cache/classification_distribution.png 了解数据分布")
    print("3. 如果类别不平衡严重，考虑使用加权损失函数")
    print("4. 如果自相关性很低，说明价格变化接近随机，模型难以学习")
