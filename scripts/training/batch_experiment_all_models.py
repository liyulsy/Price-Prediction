#!/usr/bin/env python3
"""
批量实验脚本：测试所有模型在不同配置下的性能
测试模型：TimeXer, LSTM, CNN, TimeMixer
测试配置：无GCN+无新闻, 有GCN+无新闻, 无GCN+有新闻, 有GCN+有新闻
"""

import os
import sys
import subprocess
import json
import pandas as pd
from datetime import datetime
import numpy as np

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# 模型配置
MODELS = {
    'timexer': 'scripts/training/train_timexer.py',
    'lstm': 'scripts/training/train_lstm.py', 
    'cnn': 'scripts/training/train_cnn.py',
    'timemixer': 'scripts/training/train_timemixer.py'
}

# 实验配置
EXPERIMENT_CONFIGS = [
    {"use_gcn": False, "use_news": False, "name": "baseline"},
    {"use_gcn": True, "use_news": False, "name": "gcn_only"},
    {"use_gcn": False, "use_news": True, "name": "news_only"},
    {"use_gcn": True, "use_news": True, "name": "gcn_news"},
]

# 实验参数
NUM_RUNS_PER_CONFIG = 3  # 每个配置运行3次
RESULTS_DIR = "experiments/batch_results_all_models"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def create_modified_training_script(model_name, script_path, config, run_id):
    """创建修改后的训练脚本"""
    temp_script_path = f"scripts/training/train_{model_name}_temp_{config['name']}_run{run_id}.py"
    
    # 读取原始脚本
    with open(script_path, "r") as f:
        content = f.read()
    
    # 修改配置
    content = content.replace(
        "USE_GCN = True", 
        f"USE_GCN = {config['use_gcn']}"
    )
    content = content.replace(
        "USE_NEWS_FEATURES = False", 
        f"USE_NEWS_FEATURES = {config['use_news']}"
    )
    content = content.replace(
        "USE_NEWS_FEATURES = True", 
        f"USE_NEWS_FEATURES = {config['use_news']}"
    )
    
    # 修改随机种子
    content = content.replace(
        "RANDOM_SEED = 42", 
        f"RANDOM_SEED = {42 + run_id}"
    )
    
    # 修改模型保存路径
    if "best_timexer_model.pt" in content:
        content = content.replace(
            'BEST_MODEL_NAME = "best_timexer_model.pt"',
            f'BEST_MODEL_NAME = "best_{model_name}_{config["name"]}_run{run_id}.pt"'
        )
    elif "best_lstm_model.pt" in content:
        content = content.replace(
            'BEST_MODEL_NAME = "best_lstm_model.pt"',
            f'BEST_MODEL_NAME = "best_{model_name}_{config["name"]}_run{run_id}.pt"'
        )
    elif "best_cnn_model.pt" in content:
        content = content.replace(
            'BEST_MODEL_NAME = "best_cnn_model.pt"',
            f'BEST_MODEL_NAME = "best_{model_name}_{config["name"]}_run{run_id}.pt"'
        )
    elif "best_unified_multiscale.pt" in content:
        content = content.replace(
            'BEST_MODEL_NAME = "best_unified_multiscale.pt"',
            f'BEST_MODEL_NAME = "best_{model_name}_{config["name"]}_run{run_id}.pt"'
        )
    
    # 写入临时脚本
    with open(temp_script_path, "w") as f:
        f.write(content)
    
    return temp_script_path

def run_experiment(model_name, script_path, config, run_id):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"🚀 运行实验: {model_name.upper()} - {config['name']} - Run {run_id+1}")
    print(f"   GCN: {config['use_gcn']}, News: {config['use_news']}")
    print(f"{'='*80}")
    
    # 创建修改后的训练脚本
    temp_script_path = create_modified_training_script(model_name, script_path, config, run_id)
    
    try:
        # 运行训练脚本
        result = subprocess.run(
            [sys.executable, temp_script_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 解析结果
        if result.returncode == 0:
            output_lines = result.stdout.split('\n')
            test_metrics = extract_test_metrics(output_lines)
            test_metrics['success'] = True
            test_metrics['error'] = None
            print(f"✅ 实验成功完成")
        else:
            print(f"❌ 实验失败: {result.stderr[:200]}...")
            test_metrics = {
                'success': False,
                'error': result.stderr[:500],
                'mae': None, 'mse': None, 'rmse': None, 'r2': None, 'mape': None,
                'new_mae': None, 'normalized_mae': None, 'normalized_mse': None, 'normalized_rmse': None
            }
    
    except subprocess.TimeoutExpired:
        print(f"⏰ 实验超时")
        test_metrics = {
            'success': False, 'error': 'Timeout',
            'mae': None, 'mse': None, 'rmse': None, 'r2': None, 'mape': None,
            'new_mae': None, 'normalized_mae': None, 'normalized_mse': None, 'normalized_rmse': None
        }
    
    except Exception as e:
        print(f"❌ 实验异常: {e}")
        test_metrics = {
            'success': False, 'error': str(e),
            'mae': None, 'mse': None, 'rmse': None, 'r2': None, 'mape': None,
            'new_mae': None, 'normalized_mae': None, 'normalized_mse': None, 'normalized_rmse': None
        }
    
    finally:
        # 清理临时脚本
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    
    return test_metrics

def extract_test_metrics(output_lines):
    """从输出中提取测试指标"""
    metrics = {}

    for line in output_lines:
        line = line.strip()
        # 原有指标
        if "- MAE:" in line:
            try: metrics['mae'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- MSE:" in line:
            try: metrics['mse'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- RMSE:" in line:
            try: metrics['rmse'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- R2:" in line:
            try: metrics['r2'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- MAPE:" in line:
            try: metrics['mape'] = float(line.split(":")[-1].strip())
            except: pass
        # 新增重点指标
        elif "- NEW_MAE:" in line:
            try: metrics['new_mae'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- NORMALIZED_MAE:" in line:
            try: metrics['normalized_mae'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- NORMALIZED_MSE:" in line:
            try: metrics['normalized_mse'] = float(line.split(":")[-1].strip())
            except: pass
        elif "- NORMALIZED_RMSE:" in line:
            try: metrics['normalized_rmse'] = float(line.split(":")[-1].strip())
            except: pass

    return metrics

def save_results(all_results):
    """保存实验结果"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 保存详细结果
    results_file = os.path.join(RESULTS_DIR, f"all_models_experiment_{TIMESTAMP}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 创建汇总表格
    summary_data = []
    for model_name, model_results in all_results.items():
        for config_name, runs in model_results.items():
            successful_runs = [r for r in runs if r['success']]
            if successful_runs:
                summary_row = {
                    'model': model_name,
                    'config': config_name,
                    'success_rate': len(successful_runs) / len(runs)
                }

                # 重点指标统计（优先级顺序）
                priority_metrics = ['new_mae', 'r2', 'mape', 'normalized_mae', 'normalized_mse', 'normalized_rmse']
                backup_metrics = ['mae', 'mse', 'rmse']

                # 处理重点指标
                for metric in priority_metrics:
                    values = [r[metric] for r in successful_runs if r.get(metric) is not None]
                    if values:
                        summary_row[f'{metric}_mean'] = np.mean(values)
                        summary_row[f'{metric}_std'] = np.std(values)
                        summary_row[f'{metric}_min'] = np.min(values)
                        summary_row[f'{metric}_max'] = np.max(values)

                # 如果没有重点指标，使用备用指标
                has_priority_metrics = any(f'{m}_mean' in summary_row for m in priority_metrics)
                if not has_priority_metrics:
                    for metric in backup_metrics:
                        values = [r[metric] for r in successful_runs if r.get(metric) is not None]
                        if values:
                            summary_row[f'{metric}_mean'] = np.mean(values)
                            summary_row[f'{metric}_std'] = np.std(values)
                            summary_row[f'{metric}_min'] = np.min(values)
                            summary_row[f'{metric}_max'] = np.max(values)

                summary_data.append(summary_row)
    
    # 保存汇总表格
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(RESULTS_DIR, f"all_models_summary_{TIMESTAMP}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n📊 结果已保存:")
    print(f"   详细结果: {results_file}")
    print(f"   汇总表格: {summary_file}")
    
    return summary_df

def main():
    """主函数"""
    print(f"🔬 开始所有模型批量实验 - {TIMESTAMP}")
    print(f"📋 测试模型: {list(MODELS.keys())}")
    print(f"📋 实验配置: {len(EXPERIMENT_CONFIGS)} 个配置，每个运行 {NUM_RUNS_PER_CONFIG} 次")
    
    all_results = {}
    
    for model_name, script_path in MODELS.items():
        print(f"\n🔧 开始测试模型: {model_name.upper()}")
        model_results = {}
        
        for config in EXPERIMENT_CONFIGS:
            config_results = []
            
            for run_id in range(NUM_RUNS_PER_CONFIG):
                result = run_experiment(model_name, script_path, config, run_id)
                result['model'] = model_name
                result['config'] = config['name']
                result['run_id'] = run_id
                result['use_gcn'] = config['use_gcn']
                result['use_news'] = config['use_news']
                config_results.append(result)
            
            model_results[config['name']] = config_results
        
        all_results[model_name] = model_results
    
    # 保存和分析结果
    summary_df = save_results(all_results)
    
    # 打印汇总
    print(f"\n{'='*100}")
    print("📈 所有模型实验汇总 (重点指标)")
    print(f"{'='*100}")

    if not summary_df.empty:
        for model in MODELS.keys():
            print(f"\n🔧 {model.upper()}:")
            model_data = summary_df[summary_df['model'] == model]
            for _, row in model_data.iterrows():
                config_name = row['config']
                success_rate = row['success_rate']

                # 优先显示NEW_MAE
                if 'new_mae_mean' in row and pd.notna(row['new_mae_mean']):
                    print(f"  {config_name:12} | NEW_MAE: {row['new_mae_mean']:.4f} ± {row['new_mae_std']:.4f} | 成功率: {success_rate:.1%}")
                elif 'mae_mean' in row and pd.notna(row['mae_mean']):
                    print(f"  {config_name:12} | MAE: {row['mae_mean']:.4f} ± {row['mae_std']:.4f} | 成功率: {success_rate:.1%}")
                else:
                    print(f"  {config_name:12} | 无有效指标 | 成功率: {success_rate:.1%}")

                # 显示其他重要指标
                additional_metrics = []
                if 'r2_mean' in row and pd.notna(row['r2_mean']):
                    additional_metrics.append(f"R²: {row['r2_mean']:.4f}")
                if 'mape_mean' in row and pd.notna(row['mape_mean']):
                    additional_metrics.append(f"MAPE: {row['mape_mean']*100:.2f}%")
                if 'normalized_mae_mean' in row and pd.notna(row['normalized_mae_mean']):
                    additional_metrics.append(f"NORM_MAE: {row['normalized_mae_mean']:.4f}")

                if additional_metrics:
                    print(f"  {'':12}   {' | '.join(additional_metrics)}")

    # 添加最佳模型分析
    print(f"\n{'='*100}")
    print("🏆 最佳模型分析")
    print(f"{'='*100}")

    if not summary_df.empty:
        # 找到每种配置下的最佳模型
        for config in ['baseline', 'gcn_only', 'news_only', 'gcn_news']:
            config_data = summary_df[summary_df['config'] == config]
            if not config_data.empty:
                # 优先使用NEW_MAE，否则使用MAE
                if 'new_mae_mean' in config_data.columns and config_data['new_mae_mean'].notna().any():
                    best_row = config_data.loc[config_data['new_mae_mean'].idxmin()]
                    metric_name = 'NEW_MAE'
                    metric_value = best_row['new_mae_mean']
                elif 'mae_mean' in config_data.columns and config_data['mae_mean'].notna().any():
                    best_row = config_data.loc[config_data['mae_mean'].idxmin()]
                    metric_name = 'MAE'
                    metric_value = best_row['mae_mean']
                else:
                    continue

                print(f"📊 {config:12} | 最佳: {best_row['model'].upper()} ({metric_name}: {metric_value:.4f})")

    print(f"\n✅ 所有模型批量实验完成!")

if __name__ == "__main__":
    main()
