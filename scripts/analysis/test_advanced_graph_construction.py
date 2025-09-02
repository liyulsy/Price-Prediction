#!/usr/bin/env python3
"""
测试高级图构建方法的脚本
比较不同图构建方法的效果
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

from dataloader.gnn_loader import (
    generate_edge_index, 
    generate_advanced_edge_index,
    analyze_graph_properties
)

# 配置
PRICE_CSV_PATH = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
SAVE_DIR = 'experiments/cache/graph_analysis'

def load_and_prepare_data():
    """加载和准备数据"""
    print("Loading price data...")
    
    # 加载数据
    price_df_raw = pd.read_csv(PRICE_CSV_PATH, index_col=0, parse_dates=True)
    
    # 重命名列
    rename_map = {f"{coin}-USDT": coin for coin in COIN_NAMES}
    price_df = price_df_raw.rename(columns=rename_map)[COIN_NAMES]
    
    print(f"Data shape: {price_df.shape}")
    print(f"Date range: {price_df.index.min()} to {price_df.index.max()}")
    
    return price_df

def test_graph_construction_methods(price_df):
    """测试不同的图构建方法"""
    print("\n=== Testing Graph Construction Methods ===")
    
    methods_to_test = [
        ('original', {'threshold': 0.6}),
        ('multi_layer', {'correlation_threshold': 0.3, 'volatility_threshold': 0.5, 'trend_threshold': 0.4}),
        ('dynamic', {'window_size': 168, 'overlap': 24}),
        ('domain_knowledge', {'coin_names': COIN_NAMES}),
        ('attention_based', {'top_k': 3, 'use_returns': True})
    ]
    
    results = {}
    
    for method_name, kwargs in methods_to_test:
        print(f"\nTesting {method_name} method...")
        
        try:
            if method_name == 'original':
                edge_index = generate_edge_index(price_df, **kwargs)
                edge_weights = None
            else:
                edge_index, edge_weights = generate_advanced_edge_index(
                    price_df, method=method_name, **kwargs
                )
            
            # 分析图属性
            properties = analyze_graph_properties(edge_index, edge_weights, len(COIN_NAMES))
            
            results[method_name] = {
                'edge_index': edge_index,
                'edge_weights': edge_weights,
                'properties': properties
            }
            
            print(f"  Nodes: {properties['num_nodes']}")
            print(f"  Edges: {properties['num_edges']}")
            print(f"  Density: {properties['density']:.4f}")
            print(f"  Avg degree: {properties['avg_degree']:.2f}")
            
            if edge_weights is not None:
                print(f"  Avg edge weight: {properties['avg_edge_weight']:.4f}")
            
        except Exception as e:
            print(f"  Error in {method_name}: {str(e)}")
            results[method_name] = None
    
    return results

def visualize_graphs(results):
    """可视化不同的图结构"""
    print("\n=== Visualizing Graph Structures ===")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 创建图比较表
    comparison_data = []
    
    for method_name, result in results.items():
        if result is not None:
            props = result['properties']
            comparison_data.append({
                'Method': method_name,
                'Nodes': props['num_nodes'],
                'Edges': props['num_edges'],
                'Density': props['density'],
                'Avg Degree': props['avg_degree'],
                'Max Degree': props['max_degree'],
                'Min Degree': props['min_degree'],
                'Degree Std': props['degree_std'],
                'Avg Edge Weight': props.get('avg_edge_weight', 'N/A')
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存比较表
    comparison_file = os.path.join(SAVE_DIR, f"graph_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Graph comparison saved to: {comparison_file}")
    
    # 打印比较表
    print("\nGraph Comparison:")
    print(comparison_df.to_string(index=False))
    
    # 可视化图属性
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 边数比较
    axes[0, 0].bar(comparison_df['Method'], comparison_df['Edges'])
    axes[0, 0].set_title('Number of Edges')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 密度比较
    axes[0, 1].bar(comparison_df['Method'], comparison_df['Density'])
    axes[0, 1].set_title('Graph Density')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 平均度比较
    axes[1, 0].bar(comparison_df['Method'], comparison_df['Avg Degree'])
    axes[1, 0].set_title('Average Degree')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 度标准差比较
    axes[1, 1].bar(comparison_df['Method'], comparison_df['Degree Std'])
    axes[1, 1].set_title('Degree Standard Deviation')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图像
    plot_file = os.path.join(SAVE_DIR, f"graph_properties_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Graph properties plot saved to: {plot_file}")
    
    plt.show()

def analyze_edge_weights(results):
    """分析边权重分布"""
    print("\n=== Analyzing Edge Weight Distributions ===")
    
    methods_with_weights = {name: result for name, result in results.items() 
                          if result is not None and result['edge_weights'] is not None}
    
    if not methods_with_weights:
        print("No methods with edge weights found.")
        return
    
    fig, axes = plt.subplots(1, len(methods_with_weights), figsize=(5*len(methods_with_weights), 4))
    
    if len(methods_with_weights) == 1:
        axes = [axes]
    
    for i, (method_name, result) in enumerate(methods_with_weights.items()):
        weights = result['edge_weights'].numpy()
        
        axes[i].hist(weights, bins=20, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{method_name}\nEdge Weight Distribution')
        axes[i].set_xlabel('Edge Weight')
        axes[i].set_ylabel('Frequency')
        
        # 添加统计信息
        axes[i].axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.3f}')
        axes[i].axvline(np.median(weights), color='green', linestyle='--', label=f'Median: {np.median(weights):.3f}')
        axes[i].legend()
    
    plt.tight_layout()
    
    # 保存图像
    weight_plot_file = os.path.join(SAVE_DIR, f"edge_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(weight_plot_file, dpi=300, bbox_inches='tight')
    print(f"Edge weight distributions saved to: {weight_plot_file}")
    
    plt.show()

def create_adjacency_matrices(results):
    """创建邻接矩阵可视化"""
    print("\n=== Creating Adjacency Matrix Visualizations ===")
    
    num_methods = len([r for r in results.values() if r is not None])
    fig, axes = plt.subplots(1, num_methods, figsize=(4*num_methods, 4))
    
    if num_methods == 1:
        axes = [axes]
    
    method_idx = 0
    for method_name, result in results.items():
        if result is None:
            continue
            
        # 创建邻接矩阵
        adj_matrix = np.zeros((len(COIN_NAMES), len(COIN_NAMES)))
        edge_index = result['edge_index']
        edge_weights = result['edge_weights']
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            weight = edge_weights[i].item() if edge_weights is not None else 1.0
            adj_matrix[src, dst] = weight
        
        # 可视化
        im = axes[method_idx].imshow(adj_matrix, cmap='viridis', aspect='auto')
        axes[method_idx].set_title(f'{method_name}')
        axes[method_idx].set_xticks(range(len(COIN_NAMES)))
        axes[method_idx].set_yticks(range(len(COIN_NAMES)))
        axes[method_idx].set_xticklabels(COIN_NAMES)
        axes[method_idx].set_yticklabels(COIN_NAMES)
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[method_idx])
        
        method_idx += 1
    
    plt.tight_layout()
    
    # 保存图像
    adj_plot_file = os.path.join(SAVE_DIR, f"adjacency_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(adj_plot_file, dpi=300, bbox_inches='tight')
    print(f"Adjacency matrices saved to: {adj_plot_file}")
    
    plt.show()

def main():
    """主函数"""
    print("=== Advanced Graph Construction Analysis ===")
    print(f"Analyzing coins: {COIN_NAMES}")
    
    # 创建保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 加载数据
    price_df = load_and_prepare_data()
    
    # 测试图构建方法
    results = test_graph_construction_methods(price_df)
    
    # 可视化结果
    visualize_graphs(results)
    analyze_edge_weights(results)
    create_adjacency_matrices(results)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {SAVE_DIR}")
    
    # 推荐最佳方法
    print("\n=== Recommendations ===")
    print("Based on the analysis:")
    print("1. 'multi_layer' method provides rich edge information with multiple similarity measures")
    print("2. 'dynamic' method captures time-varying relationships")
    print("3. 'domain_knowledge' method incorporates crypto-specific relationships")
    print("4. 'attention_based' method focuses on most relevant connections")
    print("\nConsider testing these methods in your training pipeline!")

if __name__ == "__main__":
    main()
