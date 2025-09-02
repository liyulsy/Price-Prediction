#!/usr/bin/env python3
"""
验证所有训练脚本的MAE修改
检查所有修改的文件是否包含正确的修改
"""

import os
import sys

def check_file_modifications(file_path):
    """检查单个文件的修改"""
    print(f"\n=== 检查文件: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查必要的导入
    has_csv_import = 'import csv' in content
    has_datetime_import = 'from datetime import datetime' in content
    
    # 检查保存函数
    has_save_function = 'def save_test_predictions(' in content
    
    # 检查新的MAE计算
    has_new_mae_calc = 'total_true_sum / total_pred_sum' in content
    has_new_mae_key = "'new_mae'" in content
    
    # 检查返回值修改
    has_modified_return = 'return metrics, all_preds, all_targets' in content
    
    # 检查测试部分的保存调用
    has_save_call = 'save_test_predictions(' in content
    
    print(f"  ✅ CSV导入: {'是' if has_csv_import else '❌ 否'}")
    print(f"  ✅ DateTime导入: {'是' if has_datetime_import else '❌ 否'}")
    print(f"  ✅ 保存函数: {'是' if has_save_function else '❌ 否'}")
    print(f"  ✅ 新MAE计算: {'是' if has_new_mae_calc else '❌ 否'}")
    print(f"  ✅ 新MAE键: {'是' if has_new_mae_key else '❌ 否'}")
    print(f"  ✅ 修改返回值: {'是' if has_modified_return else '❌ 否'}")
    print(f"  ✅ 保存调用: {'是' if has_save_call else '❌ 否'}")
    
    all_checks = [
        has_csv_import, has_datetime_import, has_save_function,
        has_new_mae_calc, has_new_mae_key, has_modified_return, has_save_call
    ]
    
    success_rate = sum(all_checks) / len(all_checks) * 100
    print(f"  📊 修改完成度: {success_rate:.1f}%")
    
    return all(all_checks)

def main():
    """主验证函数"""
    print("开始验证所有训练脚本的MAE修改...")
    
    # 要检查的文件列表
    files_to_check = [
        'scripts/training/train_timexer.py',
        'scripts/training/unified_train_script.py',
        'scripts/training/train_multiscale.py',
        'scripts/training/unified_lstm_train_script.py'
    ]
    
    results = {}
    
    for file_path in files_to_check:
        results[file_path] = check_file_modifications(file_path)
    
    print(f"\n=== 验证总结 ===")
    all_success = True
    for file_path, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{os.path.basename(file_path)}: {status}")
        if not success:
            all_success = False
    
    print(f"\n=== 整体状态 ===")
    if all_success:
        print("🎉 所有文件修改完成！")
        print("\n📋 修改内容总结:")
        print("1. ✅ 添加了CSV和datetime导入")
        print("2. ✅ 添加了save_test_predictions函数")
        print("3. ✅ 修改了MAE计算方式，添加new_mae键")
        print("4. ✅ 修改了evaluate_model返回值")
        print("5. ✅ 在测试部分添加了保存预测结果的调用")
        
        print(f"\n📁 预测结果保存位置:")
        print("experiments/cache/test_predictions/")
        print("  - test_predictions_YYYYMMDD_HHMMSS.csv")
        print("  - test_statistics_YYYYMMDD_HHMMSS.csv")
        
        print(f"\n📊 新的指标结构:")
        print("metrics = {")
        print("    'mae': <原来的MAE>,      # 平均绝对误差")
        print("    'new_mae': <新的MAE>,    # 真实值之和/预测值之和")
        print("    'mse': <MSE>,")
        print("    'rmse': <RMSE>,")
        print("    'r2': <R²>,")
        print("    # ... 其他指标")
        print("}")
    else:
        print("❌ 部分文件修改不完整，请检查上述详细信息")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
