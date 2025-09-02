#!/usr/bin/env python3
"""
批量运行所有统一的训练脚本并比较结果
"""

import subprocess
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# 配置
MODELS_TO_RUN = [
    'train_timemixer.py',
    'train_timexer.py', 
    'train_cnn.py',
    'train_lstm.py'
]

RESULTS_DIR = "experiments/cache/unified_comparison"

def run_training_script(script_name):
    """运行单个训练脚本"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {script_name}")
    print(f"{'='*60}")
    
    script_path = os.path.join(current_dir, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False, None
    
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=project_root)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully in {duration:.1f}s")
            return True, {
                'script': script_name,
                'success': True,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False, {
                'script': script_name,
                'success': False,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Exception running {script_name}: {str(e)}")
        return False, {
            'script': script_name,
            'success': False,
            'duration': duration,
            'error': str(e)
        }

def extract_metrics_from_output(output_text, script_name):
    """从输出中提取指标"""
    metrics = {
        'model': script_name.replace('train_', '').replace('.py', ''),
        'script': script_name
    }
    
    lines = output_text.split('\n')
    
    # 查找测试结果部分
    in_test_results = False
    for line in lines:
        line = line.strip()
        
        if "✅ Test Results:" in line or "Test Results:" in line:
            in_test_results = True
            continue
            
        if in_test_results and line.startswith("- "):
            # 解析指标行，格式如: "- MAE: 0.1234"
            if ":" in line:
                metric_name = line.split(":")[0].replace("- ", "").strip().lower()
                try:
                    metric_value = float(line.split(":")[1].strip())
                    metrics[metric_name] = metric_value
                except:
                    pass
        
        # 如果遇到其他部分，停止解析
        if in_test_results and ("---" in line and "Per-Coin" in line):
            break
    
    return metrics

def compare_results(all_results):
    """比较所有模型的结果"""
    print(f"\n{'='*80}")
    print("📊 MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 提取指标
    metrics_list = []
    for result in all_results:
        if result['success']:
            metrics = extract_metrics_from_output(result['stdout'], result['script'])
            metrics['duration'] = result['duration']
            metrics_list.append(metrics)
        else:
            print(f"❌ {result['script']} failed, skipping from comparison")
    
    if not metrics_list:
        print("❌ No successful runs to compare")
        return
    
    # 创建比较表
    comparison_df = pd.DataFrame(metrics_list)
    
    # 保存详细结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detailed_file = os.path.join(RESULTS_DIR, f"detailed_comparison_{timestamp}.csv")
    comparison_df.to_csv(detailed_file, index=False)
    
    # 显示主要指标比较
    key_metrics = ['mae', 'new_mae', 'r2', 'mse', 'mape']
    available_metrics = [m for m in key_metrics if m in comparison_df.columns]
    
    if available_metrics:
        print("\n🏆 RANKING BY KEY METRICS:")
        print("-" * 60)
        
        for metric in available_metrics:
            print(f"\n📈 {metric.upper()} Results:")
            if metric in ['mae', 'mse', 'mape']:  # Lower is better
                sorted_df = comparison_df.sort_values(metric)
                print("   (Lower is better)")
            elif metric == 'new_mae':  # Closer to 1.0 is better
                comparison_df['new_mae_diff'] = abs(comparison_df[metric] - 1.0)
                sorted_df = comparison_df.sort_values('new_mae_diff')
                print("   (Closer to 1.0 is better)")
            else:  # Higher is better (r2)
                sorted_df = comparison_df.sort_values(metric, ascending=False)
                print("   (Higher is better)")
            
            for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
                model_name = row['model'].upper()
                value = row[metric]
                duration = row.get('duration', 0)
                print(f"   {i}. {model_name:12} {value:8.6f} ({duration:.1f}s)")
    
    # 综合排名
    print(f"\n🎯 OVERALL PERFORMANCE SUMMARY:")
    print("-" * 60)
    
    # 计算综合得分（简单平均排名）
    ranking_scores = {}
    
    for _, row in comparison_df.iterrows():
        model = row['model']
        ranking_scores[model] = []
    
    # 为每个指标计算排名
    for metric in available_metrics:
        if metric in ['mae', 'mse', 'mape']:
            sorted_models = comparison_df.sort_values(metric)['model'].tolist()
        elif metric == 'new_mae':
            comparison_df['new_mae_diff'] = abs(comparison_df[metric] - 1.0)
            sorted_models = comparison_df.sort_values('new_mae_diff')['model'].tolist()
        else:  # r2
            sorted_models = comparison_df.sort_values(metric, ascending=False)['model'].tolist()
        
        for i, model in enumerate(sorted_models):
            ranking_scores[model].append(i + 1)  # 排名从1开始
    
    # 计算平均排名
    avg_rankings = {}
    for model, ranks in ranking_scores.items():
        if ranks:
            avg_rankings[model] = np.mean(ranks)
    
    # 按平均排名排序
    sorted_models = sorted(avg_rankings.items(), key=lambda x: x[1])
    
    print("\nOverall Ranking (based on average rank across all metrics):")
    for i, (model, avg_rank) in enumerate(sorted_models, 1):
        model_row = comparison_df[comparison_df['model'] == model].iloc[0]
        duration = model_row.get('duration', 0)
        print(f"   {i}. {model.upper():12} (avg rank: {avg_rank:.1f}, time: {duration:.1f}s)")
    
    # 推荐最佳模型
    if sorted_models:
        best_model = sorted_models[0][0]
        print(f"\n🏆 RECOMMENDED MODEL: {best_model.upper()}")
        
        best_row = comparison_df[comparison_df['model'] == best_model].iloc[0]
        print(f"   Performance highlights:")
        for metric in available_metrics:
            if metric in best_row:
                print(f"   - {metric.upper()}: {best_row[metric]:.6f}")
    
    print(f"\n📁 Detailed results saved to: {detailed_file}")
    
    # 保存简化的排名结果
    ranking_file = os.path.join(RESULTS_DIR, f"model_ranking_{timestamp}.csv")
    ranking_data = []
    for i, (model, avg_rank) in enumerate(sorted_models, 1):
        model_row = comparison_df[comparison_df['model'] == model].iloc[0]
        ranking_data.append({
            'rank': i,
            'model': model,
            'avg_rank': avg_rank,
            'duration': model_row.get('duration', 0),
            **{metric: model_row.get(metric, 0) for metric in available_metrics}
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_df.to_csv(ranking_file, index=False)
    print(f"📁 Model ranking saved to: {ranking_file}")

def main():
    """主函数"""
    print("🚀 Starting Unified Model Comparison")
    print(f"Models to run: {MODELS_TO_RUN}")
    
    all_results = []
    successful_runs = 0
    
    start_time = time.time()
    
    # 运行所有模型
    for script in MODELS_TO_RUN:
        success, result = run_training_script(script)
        all_results.append(result)
        if success:
            successful_runs += 1
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"📋 EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total scripts: {len(MODELS_TO_RUN)}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {len(MODELS_TO_RUN) - successful_runs}")
    print(f"Total time: {total_time:.1f}s")
    
    # 比较结果
    if successful_runs > 0:
        compare_results(all_results)
    else:
        print("❌ No successful runs to compare")
    
    print(f"\n✅ Unified model comparison completed!")

if __name__ == "__main__":
    main()
