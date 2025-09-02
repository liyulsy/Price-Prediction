#!/usr/bin/env python3
"""
TimeXer四实验对比脚本
一次性运行四个实验：Baseline, GCN Only, News Only, GCN+News
自动保存结果并分析对比
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

def run_experiment(use_gcn, use_news, exp_name):
    """运行单个实验"""
    
    print(f"\n🔬 实验 {exp_name}")
    print(f"   GCN: {'✅' if use_gcn else '❌'} | News: {'✅' if use_news else '❌'}")
    
    # 读取原始脚本
    script_path = "scripts/training/train_timexer.py"
    with open(script_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    try:
        # 修改配置
        modified_content = original_content
        
        # 替换GCN和新闻配置
        modified_content = modified_content.replace("USE_GCN = True", f"USE_GCN = {use_gcn}")
        modified_content = modified_content.replace("USE_GCN = False", f"USE_GCN = {use_gcn}")
        modified_content = modified_content.replace("USE_NEWS_FEATURES = True", f"USE_NEWS_FEATURES = {use_news}")
        modified_content = modified_content.replace("USE_NEWS_FEATURES = False", f"USE_NEWS_FEATURES = {use_news}")
        
        # 快速训练配置
        modified_content = modified_content.replace("EPOCHS = 50", "EPOCHS = 15")
        modified_content = modified_content.replace("EARLY_STOPPING_PATIENCE = 20", "EARLY_STOPPING_PATIENCE = 6")
        
        # 修改模型保存名称
        modified_content = modified_content.replace(
            'BEST_MODEL_NAME = "best_timexer_model.pt"',
            f'BEST_MODEL_NAME = "best_timexer_{exp_name}.pt"'
        )
        
        # 写入修改后的脚本
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # 运行实验
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=1200  # 20分钟超时
        )
        duration = time.time() - start_time
        
        # 解析结果
        if result.returncode == 0:
            metrics = extract_metrics(result.stdout)
            print(f"✅ 完成 | MAE: {metrics.get('mae', 'N/A'):.4f if isinstance(metrics.get('mae'), (int, float)) else 'N/A'} | 耗时: {duration:.1f}s")
            return {
                'name': exp_name,
                'success': True,
                'metrics': metrics,
                'duration': duration,
                'config': {'use_gcn': use_gcn, 'use_news': use_news}
            }
        else:
            print(f"❌ 失败: {result.stderr[:100]}...")
            return {
                'name': exp_name,
                'success': False,
                'error': result.stderr[:300],
                'config': {'use_gcn': use_gcn, 'use_news': use_news}
            }
    
    except subprocess.TimeoutExpired:
        print(f"⏰ 超时")
        return {'name': exp_name, 'success': False, 'error': 'Timeout', 'config': {'use_gcn': use_gcn, 'use_news': use_news}}
    except Exception as e:
        print(f"❌ 异常: {e}")
        return {'name': exp_name, 'success': False, 'error': str(e), 'config': {'use_gcn': use_gcn, 'use_news': use_news}}
    finally:
        # 恢复原始脚本
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(original_content)

def extract_metrics(output):
    """从输出中提取指标"""
    metrics = {}
    lines = output.split('\n')

    for line in lines:
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

def analyze_results(results):
    """分析和保存结果"""
    
    print(f"\n{'='*80}")
    print("📊 TimeXer四实验对比结果")
    print(f"{'='*80}")
    
    # 筛选成功的实验
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ 成功: {len(successful)}/4")
    if failed:
        print(f"❌ 失败: {len(failed)}/4")
        for f in failed:
            print(f"  - {f['name']}: {f.get('error', 'Unknown')[:50]}...")
    
    if len(successful) == 0:
        print("❌ 没有成功的实验")
        return
    
    # 创建对比表格
    table_data = []
    for result in successful:
        metrics = result['metrics']
        config = result['config']
        
        row = {
            'Experiment': result['name'],
            'GCN': '✅' if config['use_gcn'] else '❌',
            'News': '✅' if config['use_news'] else '❌',
            'Duration(s)': f"{result.get('duration', 0):.1f}",
        }
        
        # 添加指标 - 重点关注的指标
        priority_metrics = ['new_mae', 'r2', 'mape', 'normalized_mae', 'normalized_mse', 'normalized_rmse']
        for metric in priority_metrics:
            if metric in metrics:
                if metric == 'mape':
                    row[f'{metric.upper()}(%)'] = f"{metrics[metric]*100:.2f}"
                elif metric == 'new_mae':
                    row['NEW_MAE'] = f"{metrics[metric]:.4f}"
                elif metric == 'normalized_mae':
                    row['NORM_MAE'] = f"{metrics[metric]:.4f}"
                elif metric == 'normalized_mse':
                    row['NORM_MSE'] = f"{metrics[metric]:.4f}"
                elif metric == 'normalized_rmse':
                    row['NORM_RMSE'] = f"{metrics[metric]:.4f}"
                else:
                    row[metric.upper()] = f"{metrics[metric]:.4f}"
            else:
                if metric == 'mape':
                    row[f'{metric.upper()}(%)'] = "N/A"
                elif metric == 'new_mae':
                    row['NEW_MAE'] = "N/A"
                elif metric == 'normalized_mae':
                    row['NORM_MAE'] = "N/A"
                elif metric == 'normalized_mse':
                    row['NORM_MSE'] = "N/A"
                elif metric == 'normalized_rmse':
                    row['NORM_RMSE'] = "N/A"
                else:
                    row[metric.upper()] = "N/A"
        
        table_data.append(row)
    
    # 打印表格
    df = pd.DataFrame(table_data)
    print(f"\n📋 实验对比表:")
    print(df.to_string(index=False))
    
    # 分析最佳结果
    if len(successful) > 1:
        print(f"\n🏆 最佳结果:")

        # NEW_MAE最低（优先指标）
        new_mae_results = [(r['name'], r['metrics']['new_mae']) for r in successful if 'new_mae' in r['metrics']]
        if new_mae_results:
            best_new_mae = min(new_mae_results, key=lambda x: x[1])
            print(f"   最低NEW_MAE: {best_new_mae[0]} ({best_new_mae[1]:.4f})")
        else:
            # 如果没有NEW_MAE，使用普通MAE
            mae_results = [(r['name'], r['metrics']['mae']) for r in successful if 'mae' in r['metrics']]
            if mae_results:
                best_mae = min(mae_results, key=lambda x: x[1])
                print(f"   最低MAE: {best_mae[0]} ({best_mae[1]:.4f})")

        # R²最高
        r2_results = [(r['name'], r['metrics']['r2']) for r in successful if 'r2' in r['metrics']]
        if r2_results:
            best_r2 = max(r2_results, key=lambda x: x[1])
            print(f"   最高R²:  {best_r2[0]} ({best_r2[1]:.4f})")

        # 最低MAPE
        mape_results = [(r['name'], r['metrics']['mape']) for r in successful if 'mape' in r['metrics']]
        if mape_results:
            best_mape = min(mape_results, key=lambda x: x[1])
            print(f"   最低MAPE: {best_mape[0]} ({best_mape[1]*100:.2f}%)")

        # 最低NORMALIZED_MAE
        norm_mae_results = [(r['name'], r['metrics']['normalized_mae']) for r in successful if 'normalized_mae' in r['metrics']]
        if norm_mae_results:
            best_norm_mae = min(norm_mae_results, key=lambda x: x[1])
            print(f"   最低NORM_MAE: {best_norm_mae[0]} ({best_norm_mae[1]:.4f})")
    
    # 组件贡献分析
    baseline = next((r for r in successful if r['name'] == 'baseline'), None)
    gcn_only = next((r for r in successful if r['name'] == 'gcn_only'), None)
    news_only = next((r for r in successful if r['name'] == 'news_only'), None)
    gcn_news = next((r for r in successful if r['name'] == 'gcn_news'), None)

    # 优先使用NEW_MAE进行分析
    metric_key = 'new_mae' if baseline and 'new_mae' in baseline['metrics'] else 'mae'
    metric_name = 'NEW_MAE' if metric_key == 'new_mae' else 'MAE'

    if baseline and metric_key in baseline['metrics']:
        baseline_value = baseline['metrics'][metric_key]
        print(f"\n🔍 相对基线改善 (基线{metric_name}: {baseline_value:.4f}):")

        for exp, name in [(gcn_only, 'GCN'), (news_only, 'News'), (gcn_news, 'GCN+News')]:
            if exp and metric_key in exp['metrics']:
                exp_value = exp['metrics'][metric_key]
                improvement = (baseline_value - exp_value) / baseline_value * 100
                status = '改善' if improvement > 0 else '下降'
                print(f"   {name:8}: {improvement:+6.2f}% ({status})")

    # 多指标分析
    print(f"\n📊 多指标对比分析:")
    if baseline:
        for metric, display_name in [('new_mae', 'NEW_MAE'), ('r2', 'R²'), ('mape', 'MAPE'),
                                   ('normalized_mae', 'NORM_MAE'), ('normalized_mse', 'NORM_MSE')]:
            if metric in baseline['metrics']:
                baseline_val = baseline['metrics'][metric]
                gcn_news_val = gcn_news['metrics'].get(metric) if gcn_news else None

                if gcn_news_val is not None:
                    if metric in ['new_mae', 'mape', 'normalized_mae', 'normalized_mse']:
                        # 越小越好的指标
                        improvement = (baseline_val - gcn_news_val) / baseline_val * 100
                        comparison = "✅ 改善" if improvement > 0 else "❌ 下降"
                    else:
                        # 越大越好的指标 (R²)
                        improvement = (gcn_news_val - baseline_val) / baseline_val * 100
                        comparison = "✅ 改善" if improvement > 0 else "❌ 下降"

                    print(f"   {display_name:10}: {improvement:+6.2f}% ({comparison})")

    # 检查预期结果
    if gcn_news and baseline:
        check_metric = 'new_mae' if 'new_mae' in gcn_news['metrics'] and 'new_mae' in baseline['metrics'] else 'mae'

        if check_metric in gcn_news['metrics'] and check_metric in baseline['metrics']:
            gcn_news_value = gcn_news['metrics'][check_metric]
            baseline_value = baseline['metrics'][check_metric]
            metric_display = 'NEW_MAE' if check_metric == 'new_mae' else 'MAE'

            if gcn_news_value < baseline_value:
                print(f"\n✅ 结果符合预期: GCN+News {metric_display} ({gcn_news_value:.4f}) < Baseline {metric_display} ({baseline_value:.4f})")
            else:
                print(f"\n⚠️  结果不符合预期: GCN+News {metric_display} ({gcn_news_value:.4f}) >= Baseline {metric_display} ({baseline_value:.4f})")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiments/four_experiments"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存CSV
    csv_file = os.path.join(results_dir, f"comparison_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    # 保存JSON
    json_file = os.path.join(results_dir, f"detailed_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存:")
    print(f"   📊 对比表格: {csv_file}")
    print(f"   📋 详细结果: {json_file}")

def main():
    """主函数"""
    print("🚀 TimeXer四实验对比")
    print("⚡ 快速版本 - 每个实验15轮训练")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 四个实验配置
    experiments = [
        ("baseline", False, False, "基线（仅价格）"),
        ("gcn_only", True, False, "仅GCN"),
        ("news_only", False, True, "仅新闻"),
        ("gcn_news", True, True, "GCN+新闻"),
    ]
    
    results = []
    total_start = time.time()
    
    # 运行所有实验
    for exp_name, use_gcn, use_news, desc in experiments:
        print(f"\n{'='*50}")
        print(f"📋 {desc}")
        result = run_experiment(use_gcn, use_news, exp_name)
        results.append(result)
    
    total_time = time.time() - total_start
    print(f"\n⏱️  总耗时: {total_time:.1f} 秒")
    
    # 分析结果
    analyze_results(results)
    
    print(f"\n🎉 四实验对比完成！")
    print(f"📁 查看 experiments/four_experiments/ 目录获取详细结果")

if __name__ == "__main__":
    main()
