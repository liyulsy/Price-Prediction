#!/usr/bin/env python3
"""
验证价格变化的随机性
测试不同的预测目标和时间窗口
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import sys

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']

def test_predictability():
    """测试不同时间窗口和阈值的可预测性"""
    
    # 加载数据
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df = price_df_raw.rename(columns=rename_map)[COIN_NAMES]
    
    results = []
    
    # 测试不同的预测设置
    test_configs = [
        {'window': 1, 'threshold': 0, 'name': '1小时涨跌'},
        {'window': 4, 'threshold': 0, 'name': '4小时涨跌'},
        {'window': 24, 'threshold': 0, 'name': '24小时涨跌'},
        {'window': 1, 'threshold': 0.02, 'name': '1小时大涨跌(>2%)'},
        {'window': 4, 'threshold': 0.02, 'name': '4小时大涨跌(>2%)'},
        {'window': 24, 'threshold': 0.02, 'name': '24小时大涨跌(>2%)'},
    ]
    
    for config in test_configs:
        print(f"\n🧪 测试: {config['name']}")
        print("=" * 50)
        
        window = config['window']
        threshold = config['threshold']
        
        # 计算目标变量
        if window == 1:
            returns = price_df.pct_change().dropna()
        else:
            returns = price_df.pct_change(periods=window).dropna()
        
        # 应用阈值
        if threshold > 0:
            # 三分类：大涨、大跌、横盘
            targets = {}
            for coin in COIN_NAMES:
                coin_returns = returns[coin]
                targets[coin] = np.where(coin_returns > threshold, 1,  # 大涨
                                       np.where(coin_returns < -threshold, -1, 0))  # 大跌 vs 横盘
            class_names = ['大跌', '横盘', '大涨']
        else:
            # 二分类：涨跌
            targets = {}
            for coin in COIN_NAMES:
                targets[coin] = (returns[coin] > 0).astype(int)
            class_names = ['跌', '涨']
        
        # 为每个币种测试
        coin_results = []
        for coin in COIN_NAMES:
            coin_returns = returns[coin].values
            coin_targets = targets[coin]
            
            # 创建简单特征（过去几个时间点的收益率）
            feature_window = min(24, len(coin_returns) - window)
            X = []
            y = []
            
            for i in range(feature_window, len(coin_returns) - window):
                # 特征：过去24小时的收益率
                features = coin_returns[i-feature_window:i]
                if i + window - 1 < len(coin_targets):
                    target = coin_targets[i + window - 1]
                    X.append(features)
                    y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            if len(np.unique(y)) < 2:
                print(f"  {coin}: 跳过（类别不足）")
                continue
            
            # 训练测试分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 训练随机森林
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # 基线准确率（预测多数类）
            baseline = max(np.bincount(y_test)) / len(y_test)
            
            coin_results.append({
                'coin': coin,
                'accuracy': accuracy,
                'baseline': baseline,
                'improvement': accuracy - baseline,
                'samples': len(y_test)
            })
            
            print(f"  {coin}: 准确率={accuracy:.3f}, 基线={baseline:.3f}, 提升={accuracy-baseline:+.3f}")
        
        # 计算平均结果
        if coin_results:
            avg_accuracy = np.mean([r['accuracy'] for r in coin_results])
            avg_baseline = np.mean([r['baseline'] for r in coin_results])
            avg_improvement = avg_accuracy - avg_baseline
            
            print(f"\n  📊 平均结果:")
            print(f"    准确率: {avg_accuracy:.3f}")
            print(f"    基线: {avg_baseline:.3f}")
            print(f"    提升: {avg_improvement:+.3f}")
            
            if avg_improvement > 0.05:
                print(f"    ✅ 有一定可预测性")
            elif avg_improvement > 0.02:
                print(f"    ⚠️  可预测性较弱")
            else:
                print(f"    ❌ 基本无法预测（接近随机）")
            
            results.append({
                'config': config['name'],
                'avg_accuracy': avg_accuracy,
                'avg_baseline': avg_baseline,
                'avg_improvement': avg_improvement
            })
    
    # 总结最佳配置
    print(f"\n🏆 最佳预测配置排名:")
    print("=" * 60)
    results.sort(key=lambda x: x['avg_improvement'], reverse=True)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['config']}: "
              f"准确率={result['avg_accuracy']:.3f}, "
              f"提升={result['avg_improvement']:+.3f}")
    
    return results

if __name__ == '__main__':
    print("🚀 开始可预测性验证...")
    results = test_predictability()
    print("\n✅ 验证完成！")
