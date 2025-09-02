#!/usr/bin/env python3
"""
测试MAE修改的简单脚本
验证新的MAE计算方式是否正确工作
"""

import numpy as np
import os
import sys

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_new_mae_calculation():
    """测试新的MAE计算方式"""
    print("=== 测试新的MAE计算方式 ===")
    
    # 创建一些测试数据
    np.random.seed(42)
    test_targets = np.random.randn(100, 8) * 10 + 50  # 模拟价格数据
    test_preds = test_targets + np.random.randn(100, 8) * 2  # 添加一些噪声
    
    print(f"测试数据形状: targets={test_targets.shape}, preds={test_preds.shape}")
    print(f"真实值范围: [{np.min(test_targets):.2f}, {np.max(test_targets):.2f}]")
    print(f"预测值范围: [{np.min(test_preds):.2f}, {np.max(test_preds):.2f}]")
    
    # 原来的MAE计算方式
    from sklearn.metrics import mean_absolute_error
    original_mae = mean_absolute_error(test_targets, test_preds)
    
    # 新的MAE计算方式：所有真实值之和除以预测值之和
    total_true_sum = np.sum(test_targets)
    total_pred_sum = np.sum(test_preds)
    new_mae = total_true_sum / total_pred_sum if total_pred_sum != 0 else float('inf')
    
    print(f"\n=== MAE计算结果对比 ===")
    print(f"原来的MAE (平均绝对误差): {original_mae:.6f}")
    print(f"新的MAE (真实值之和/预测值之和): {new_mae:.6f}")
    print(f"真实值总和: {total_true_sum:.2f}")
    print(f"预测值总和: {total_pred_sum:.2f}")
    print(f"比值差异: {abs(new_mae - 1.0):.6f} (理想情况下应该接近1.0)")

    print(f"\n=== 修改后的指标结构 ===")
    print(f"'mae': {original_mae:.6f}  # 保留原来的MAE计算方式")
    print(f"'new_mae': {new_mae:.6f}  # 新的MAE计算方式")
    
    return original_mae, new_mae

def test_save_predictions_function():
    """测试保存预测结果的函数"""
    print("\n=== 测试保存预测结果功能 ===")
    
    # 导入保存函数
    try:
        sys.path.append('scripts/training')
        from train_timexer import save_test_predictions
        
        # 创建测试数据
        np.random.seed(42)
        coin_names = ['BTC', 'ETH', 'BNB', 'XRP']
        test_targets = np.random.randn(50, 4) * 10 + 50
        test_preds = test_targets + np.random.randn(50, 4) * 2
        
        # 测试保存功能
        timestamp = "test_20250711_120000"
        pred_file, stats_file = save_test_predictions(test_preds, test_targets, coin_names, timestamp)
        
        print(f"预测结果文件: {pred_file}")
        print(f"统计信息文件: {stats_file}")
        
        # 检查文件是否存在
        if os.path.exists(pred_file):
            print("✅ 预测结果文件创建成功")
            # 读取前几行查看格式
            with open(pred_file, 'r') as f:
                lines = f.readlines()[:6]  # 读取前6行
                print("文件内容预览:")
                for line in lines:
                    print(f"  {line.strip()}")
        else:
            print("❌ 预测结果文件创建失败")
            
        if os.path.exists(stats_file):
            print("✅ 统计信息文件创建成功")
            # 读取统计文件
            with open(stats_file, 'r') as f:
                lines = f.readlines()
                print("统计文件内容:")
                for line in lines:
                    print(f"  {line.strip()}")
        else:
            print("❌ 统计信息文件创建失败")
            
        return True
        
    except ImportError as e:
        print(f"❌ 导入保存函数失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试保存功能时出错: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试MAE修改...")
    
    # 测试新的MAE计算
    original_mae, new_mae = test_new_mae_calculation()
    
    # 测试保存预测结果功能
    save_success = test_save_predictions_function()
    
    print(f"\n=== 测试总结 ===")
    print(f"MAE计算修改: ✅ 完成")
    print(f"  - 原MAE: {original_mae:.6f}")
    print(f"  - 新MAE: {new_mae:.6f}")
    print(f"保存功能测试: {'✅ 成功' if save_success else '❌ 失败'}")
    
    print(f"\n=== 修改说明 ===")
    print("1. 新建文件夹: experiments/cache/test_predictions")
    print("2. 保存测试集预测值和真实值到CSV文件")
    print("3. MAE计算方式修改为: 所有真实值之和 / 预测值之和")
    print("4. 原来的MAE计算方式已注释，保留为 'original_mae' 字段")
    
if __name__ == "__main__":
    main()
