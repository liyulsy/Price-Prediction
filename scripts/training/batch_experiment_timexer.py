#!/usr/bin/env python3
"""
批量实验脚本：系统性测试TimeXer模型在不同配置下的性能
测试配置：
1. 无GCN + 无新闻
2. 有GCN + 无新闻  
3. 无GCN + 有新闻
4. 有GCN + 有新闻

每个配置运行多次以评估稳定性
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

# 实验配置
EXPERIMENT_CONFIGS = [
    {"use_gcn": False, "use_news": False, "name": "baseline"},
    {"use_gcn": True, "use_news": False, "name": "gcn_only"},
    {"use_gcn": False, "use_news": True, "name": "news_only"},
    {"use_gcn": True, "use_news": True, "name": "gcn_news"},
]

# 实验参数
NUM_RUNS_PER_CONFIG = 5  # 每个配置运行5次
RESULTS_DIR = "experiments/batch_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def create_modified_training_script(config, run_id):
    """创建修改后的训练脚本"""
    script_path = f"scripts/training/train_timexer_temp_{config['name']}_run{run_id}.py"
    
    # 读取原始脚本
    with open("scripts/training/train_timexer.py", "r") as f:
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
    
    # 修改随机种子以确保每次运行都不同
    content = content.replace(
        "RANDOM_SEED = 42", 
        f"RANDOM_SEED = {42 + run_id}"
    )
    
    # 修改模型保存路径
    content = content.replace(
        'BEST_MODEL_NAME = "best_timexer_model.pt"',
        f'BEST_MODEL_NAME = "best_timexer_{config["name"]}_run{run_id}.pt"'
    )
    
    # 写入临时脚本
    with open(script_path, "w") as f:
        f.write(content)
    
    return script_path

def run_experiment(config, run_id):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"🚀 运行实验: {config['name']} - Run {run_id+1}")
    print(f"   GCN: {config['use_gcn']}, News: {config['use_news']}")
    print(f"{'='*60}")
    
    # 创建修改后的训练脚本
    script_path = create_modified_training_script(config, run_id)
    
    try:
        # 运行训练脚本
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        # 解析结果
        if result.returncode == 0:
            # 从输出中提取测试指标
            output_lines = result.stdout.split('\n')
            test_metrics = extract_test_metrics(output_lines)
            test_metrics['success'] = True
            test_metrics['error'] = None
        else:
            print(f"❌ 实验失败: {result.stderr}")
            test_metrics = {
                'success': False,
                'error': result.stderr,
                'mae': None,
                'mse': None,
                'rmse': None,
                'r2': None,
                'mape': None,
                'new_mae': None,
                'normalized_mae': None,
                'normalized_mse': None,
                'normalized_rmse': None
            }
    
    except subprocess.TimeoutExpired:
        print(f"⏰ 实验超时")
        test_metrics = {
            'success': False,
            'error': 'Timeout',
            'mae': None,
            'mse': None,
            'rmse': None,
            'r2': None,
            'mape': None,
            'new_mae': None,
            'normalized_mae': None,
            'normalized_mse': None,
            'normalized_rmse': None
        }
    
    except Exception as e:
        print(f"❌ 实验异常: {e}")
        test_metrics = {
            'success': False,
            'error': str(e),
            'mae': None,
            'mse': None,
            'rmse': None,
            'r2': None,
            'mape': None,
            'new_mae': None,
            'normalized_mae': None,
            'normalized_mse': None,
            'normalized_rmse': None
        }
    
    finally:
        # 清理临时脚本
        if os.path.exists(script_path):
            os.remove(script_path)
    
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
        # 新增指标
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
    results_file = os.path.join(RESULTS_DIR, f"batch_experiment_{TIMESTAMP}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 创建汇总表格
    summary_data = []
    for config_name, runs in all_results.items():
        successful_runs = [r for r in runs if r['success']]
        if successful_runs:
            # 重点关注的指标
            priority_metrics = ['new_mae', 'r2', 'mape', 'normalized_mae', 'normalized_mse', 'normalized_rmse']
            # 备用指标
            backup_metrics = ['mae', 'mse', 'rmse']

            summary_row = {'config': config_name}

            # 优先处理重点指标
            for metric in priority_metrics:
                values = [r[metric] for r in successful_runs if r.get(metric) is not None]
                if values:
                    summary_row[f'{metric}_mean'] = np.mean(values)
                    summary_row[f'{metric}_std'] = np.std(values)
                    summary_row[f'{metric}_min'] = np.min(values)
                    summary_row[f'{metric}_max'] = np.max(values)

            # 如果没有重点指标，使用备用指标
            if not any(f'{m}_mean' in summary_row for m in priority_metrics):
                for metric in backup_metrics:
                    values = [r[metric] for r in successful_runs if r.get(metric) is not None]
                    if values:
                        summary_row[f'{metric}_mean'] = np.mean(values)
                        summary_row[f'{metric}_std'] = np.std(values)
                        summary_row[f'{metric}_min'] = np.min(values)
                        summary_row[f'{metric}_max'] = np.max(values)

            summary_row['success_rate'] = len(successful_runs) / len(runs)
            summary_data.append(summary_row)
    
    # 保存汇总表格
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(RESULTS_DIR, f"batch_summary_{TIMESTAMP}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n📊 结果已保存:")
    print(f"   详细结果: {results_file}")
    print(f"   汇总表格: {summary_file}")
    
    return summary_df

def main():
    """主函数"""
    print(f"🔬 开始批量实验 - {TIMESTAMP}")
    print(f"📋 实验配置: {len(EXPERIMENT_CONFIGS)} 个配置，每个运行 {NUM_RUNS_PER_CONFIG} 次")
    
    all_results = {}
    
    for config in EXPERIMENT_CONFIGS:
        config_results = []
        
        for run_id in range(NUM_RUNS_PER_CONFIG):
            result = run_experiment(config, run_id)
            result['config'] = config['name']
            result['run_id'] = run_id
            result['use_gcn'] = config['use_gcn']
            result['use_news'] = config['use_news']
            config_results.append(result)
        
        all_results[config['name']] = config_results
    
    # 保存和分析结果
    summary_df = save_results(all_results)
    
    # 打印汇总
    print(f"\n{'='*80}")
    print("📈 实验汇总 (重点指标)")
    print(f"{'='*80}")

    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            config_name = row['config']
            success_rate = row['success_rate']

            # 优先显示NEW_MAE
            if 'new_mae_mean' in row and pd.notna(row['new_mae_mean']):
                print(f"{config_name:15} | NEW_MAE: {row['new_mae_mean']:.4f} ± {row['new_mae_std']:.4f} | 成功率: {success_rate:.1%}")
            elif 'mae_mean' in row and pd.notna(row['mae_mean']):
                print(f"{config_name:15} | MAE: {row['mae_mean']:.4f} ± {row['mae_std']:.4f} | 成功率: {success_rate:.1%}")
            else:
                print(f"{config_name:15} | 无有效指标 | 成功率: {success_rate:.1%}")

            # 显示其他重要指标
            additional_metrics = []
            if 'r2_mean' in row and pd.notna(row['r2_mean']):
                additional_metrics.append(f"R²: {row['r2_mean']:.4f}")
            if 'mape_mean' in row and pd.notna(row['mape_mean']):
                additional_metrics.append(f"MAPE: {row['mape_mean']*100:.2f}%")
            if 'normalized_mae_mean' in row and pd.notna(row['normalized_mae_mean']):
                additional_metrics.append(f"NORM_MAE: {row['normalized_mae_mean']:.4f}")

            if additional_metrics:
                print(f"{'':15}   {' | '.join(additional_metrics)}")

    print(f"\n✅ 批量实验完成!")

if __name__ == "__main__":
    main()
