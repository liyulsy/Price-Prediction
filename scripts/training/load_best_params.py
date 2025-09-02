#!/usr/bin/env python3
"""
最佳参数加载工具

这个脚本提供了多种方式来加载和使用贝叶斯优化得到的最佳参数。

使用方法:
    python scripts/training/load_best_params.py
    python scripts/training/load_best_params.py --format python
    python scripts/training/load_best_params.py --file path/to/best_params.json
"""

import argparse
import json
import os
import sys
import yaml
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

def load_best_params_json(filepath):
    """从JSON文件加载最佳参数"""
    try:
        with open(filepath, 'r') as f:
            params = json.load(f)
        return params
    except Exception as e:
        print(f"❌ 加载JSON参数文件失败: {e}")
        return None

def load_best_params_yaml(filepath):
    """从YAML文件加载最佳参数"""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        # 重构YAML数据为扁平的参数字典
        params = {}
        if 'wpmixer' in data:
            for key, value in data['wpmixer'].items():
                if key == 'price_seq_len':
                    params['price_seq_len'] = value
                else:
                    params[key] = value
        
        if 'mlp' in data:
            params['mlp_hidden_dim_1'] = data['mlp']['hidden_dim_1']
            params['mlp_hidden_dim_2'] = data['mlp']['hidden_dim_2']
        
        if 'training' in data:
            for key, value in data['training'].items():
                params[key] = value
        
        return params, data
    except Exception as e:
        print(f"❌ 加载YAML参数文件失败: {e}")
        return None, None

def find_latest_params_file(cache_dir="experiments/cache/bayesian_optimization", format_type="json"):
    """查找最新的参数文件"""
    if not os.path.exists(cache_dir):
        return None
    
    if format_type == "json":
        # 查找最新的best_params.json
        json_file = os.path.join(cache_dir, "best_params.json")
        if os.path.exists(json_file):
            return json_file
    
    # 查找带时间戳的文件
    files = []
    for filename in os.listdir(cache_dir):
        if format_type == "python" and filename.startswith("best_params_") and filename.endswith(".py"):
            files.append(os.path.join(cache_dir, filename))
        elif format_type == "yaml" and filename.startswith("best_params_") and filename.endswith(".yaml"):
            files.append(os.path.join(cache_dir, filename))
        elif format_type == "json" and filename.startswith("best_params_") and filename.endswith(".json"):
            files.append(os.path.join(cache_dir, filename))
    
    if files:
        # 返回最新的文件
        return max(files, key=os.path.getmtime)
    
    return None

def create_wpmixer_config_from_params(params):
    """从参数字典创建WPMixer配置对象"""
    class WPMixerConfigs:
        def __init__(self, params_dict):
            self.input_length = params_dict.get('price_seq_len', 60)
            self.pred_length = 1
            self.num_coins = 8
            self.d_model = params_dict.get('d_model', 128)
            self.patch_len = params_dict.get('patch_len', 8)
            self.patch_stride = params_dict.get('patch_stride', 4)
            self.wavelet_name = params_dict.get('wavelet_name', 'db4')
            self.level = params_dict.get('wavelet_level', 2)
            self.tfactor = params_dict.get('tfactor', 2)
            self.dfactor = params_dict.get('dfactor', 4)
            self.no_decomposition = False
            self.use_amp = False
            self.dropout = params_dict.get('dropout', 0.1)
            self.task_type = 'regression'
            self.device = 'cuda'
    
    return WPMixerConfigs(params)

def create_training_config_from_params(params):
    """从参数字典创建训练配置"""
    return {
        'batch_size': params.get('batch_size', 32),
        'learning_rate': params.get('learning_rate', 0.001),
        'weight_decay': params.get('weight_decay', 1e-4),
        'epochs': params.get('epochs', 50),
        'dropout': params.get('dropout', 0.1),
        'mlp_hidden_dim_1': params.get('mlp_hidden_dim_1', 1024),
        'mlp_hidden_dim_2': params.get('mlp_hidden_dim_2', 512)
    }

def print_params_summary(params, extra_data=None):
    """打印参数摘要"""
    print("🎯 最佳贝叶斯优化参数摘要")
    print("="*50)
    
    if extra_data and 'optimization_info' in extra_data:
        opt_info = extra_data['optimization_info']
        print(f"优化评分: {opt_info.get('score', 'N/A')}")
        print(f"优化目标: {opt_info.get('objective', 'N/A')}")
        print(f"生成时间: {opt_info.get('timestamp', 'N/A')}")
        print()
    
    print("📋 WPMixer参数:")
    wpmixer_params = ['d_model', 'patch_len', 'patch_stride', 'price_seq_len', 
                     'wavelet_name', 'wavelet_level', 'tfactor', 'dfactor']
    for param in wpmixer_params:
        if param in params:
            print(f"  {param}: {params[param]}")
    
    print("\n🏗️ MLP参数:")
    mlp_params = ['mlp_hidden_dim_1', 'mlp_hidden_dim_2']
    for param in mlp_params:
        if param in params:
            print(f"  {param}: {params[param]}")
    
    print("\n🏃 训练参数:")
    training_params = ['batch_size', 'learning_rate', 'weight_decay', 'dropout', 'epochs']
    for param in training_params:
        if param in params:
            value = params[param]
            if param in ['learning_rate', 'weight_decay']:
                print(f"  {param}: {value:.8f}")
            elif param == 'dropout':
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")
    
    if extra_data and 'test_metrics' in extra_data:
        print("\n📊 测试集性能:")
        metrics = extra_data['test_metrics']
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.6f}")

def main():
    parser = argparse.ArgumentParser(description='加载最佳贝叶斯优化参数')
    parser.add_argument('--file', type=str, help='指定参数文件路径')
    parser.add_argument('--format', choices=['json', 'yaml', 'python'], default='json',
                       help='参数文件格式 (默认: json)')
    parser.add_argument('--cache-dir', type=str, default='experiments/cache/bayesian_optimization',
                       help='缓存目录路径')
    parser.add_argument('--create-config', action='store_true',
                       help='创建配置对象示例')
    
    args = parser.parse_args()
    
    # 确定要加载的文件
    if args.file:
        filepath = args.file
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            return
    else:
        filepath = find_latest_params_file(args.cache_dir, args.format)
        if not filepath:
            print(f"❌ 在 {args.cache_dir} 中未找到 {args.format} 格式的参数文件")
            return
    
    print(f"📂 加载参数文件: {filepath}")
    
    # 加载参数
    params = None
    extra_data = None
    
    if args.format == 'json' or filepath.endswith('.json'):
        params = load_best_params_json(filepath)
    elif args.format == 'yaml' or filepath.endswith('.yaml'):
        params, extra_data = load_best_params_yaml(filepath)
    elif args.format == 'python' or filepath.endswith('.py'):
        print("💡 Python配置文件请直接导入使用:")
        print(f"   from {os.path.basename(filepath)[:-3]} import *")
        return
    
    if params is None:
        print("❌ 参数加载失败")
        return
    
    # 打印参数摘要
    print_params_summary(params, extra_data)
    
    # 创建配置对象示例
    if args.create_config:
        print("\n🔧 配置对象创建示例:")
        print("-" * 30)
        
        wpmixer_config = create_wpmixer_config_from_params(params)
        training_config = create_training_config_from_params(params)
        
        print("# WPMixer配置对象")
        print("wpmixer_config = create_wpmixer_config_from_params(params)")
        print("# 训练配置字典")
        print("training_config = create_training_config_from_params(params)")
        
        print(f"\n✅ 配置对象已创建")
        print(f"WPMixer配置: {wpmixer_config.__dict__}")
        print(f"训练配置: {training_config}")

if __name__ == '__main__':
    main()
