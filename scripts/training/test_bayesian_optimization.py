#!/usr/bin/env python3
"""
贝叶斯优化测试脚本

这个脚本用于测试贝叶斯优化的基本功能，
使用很少的迭代次数来快速验证整个流程。
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

def test_imports():
    """测试所有必要的导入"""
    print("🔍 测试导入...")
    
    try:
        import skopt
        print(f"✅ scikit-optimize: {skopt.__version__}")
    except ImportError as e:
        print(f"❌ scikit-optimize导入失败: {e}")
        return False
    
    try:
        from scripts.training.bayesian_optimize_wpmixer import (
            search_space, objective, prepare_data, WPMixerConfigs
        )
        print(f"✅ 贝叶斯优化模块导入成功")
        print(f"   搜索空间维度: {len(search_space)}")
    except ImportError as e:
        print(f"❌ 贝叶斯优化模块导入失败: {e}")
        return False
    
    try:
        from models.MixModel.unified_wpmixer import UnifiedWPMixer
        print(f"✅ UnifiedWPMixer模型导入成功")
    except ImportError as e:
        print(f"❌ UnifiedWPMixer模型导入失败: {e}")
        return False
    
    return True

def test_data_availability():
    """测试数据文件是否存在"""
    print("\n📊 测试数据可用性...")
    
    data_path = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return False
    
    print(f"✅ 数据文件存在: {data_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
    print(f"   文件大小: {file_size:.2f} MB")
    
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    
    try:
        from scripts.training.bayesian_optimize_wpmixer import WPMixerConfigs, UnifiedWPMixer, DEVICE
        
        # 创建测试配置
        configs = WPMixerConfigs(
            input_length=60,
            num_coins=8,
            d_model=64,
            patch_len=8,
            patch_stride=4,
            wavelet_name='db4',
            level=2,
            tfactor=2,
            dfactor=2,
            dropout=0.1
        )
        
        # 创建模型
        model = UnifiedWPMixer(
            configs=configs,
            use_gcn=False,
            gcn_config='improved_light',
            news_feature_dim=None,
            gcn_hidden_dim=256,
            gcn_output_dim=128,
            news_processed_dim=64,
            mlp_hidden_dim_1=512,
            mlp_hidden_dim_2=256,
            num_classes=1
        ).to(DEVICE)
        
        print(f"✅ 模型创建成功")
        print(f"   设备: {DEVICE}")
        print(f"   参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 测试前向传播
        batch_size = 4
        seq_len = 60
        num_coins = 8
        
        test_input = torch.randn(batch_size, seq_len, num_coins).to(DEVICE)
        
        with torch.no_grad():
            output = model(price_data=test_input)
            print(f"   输入形状: {test_input.shape}")
            print(f"   输出形状: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preparation():
    """测试数据准备"""
    print("\n📋 测试数据准备...")

    try:
        from scripts.training.bayesian_optimize_wpmixer import prepare_data
        import scripts.training.bayesian_optimize_wpmixer as bo_module

        # 准备数据
        result = prepare_data()

        # 检查返回值
        if result is not None:
            dataset, train_dataset, val_dataset, test_dataset, scaler = result
            print(f"✅ 数据准备成功")
            print(f"   数据集大小: {len(dataset)}")
            print(f"   币种数量: {dataset.num_coins}")
            print(f"   序列长度: {dataset.seq_len}")

            # 测试获取一个样本
            sample = dataset[0]
            print(f"   样本键: {list(sample.keys())}")
            print(f"   价格序列形状: {sample['price_seq'].shape}")
            print(f"   目标价格形状: {sample['target_price'].shape}")

            # 检查全局变量
            if bo_module.global_dataset is not None:
                print(f"   ✅ 全局变量设置成功")
            else:
                print(f"   ⚠️ 全局变量未设置，但函数返回了数据")

            return True
        else:
            print(f"❌ 数据准备失败: 返回值为None")
            return False

    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_objective_function():
    """测试目标函数"""
    print("\n🎯 测试目标函数...")

    try:
        from scripts.training.bayesian_optimize_wpmixer import (
            objective, param_names, OPTIMIZATION_OBJECTIVE, calculate_optimization_score
        )

        print(f"   当前优化目标: {OPTIMIZATION_OBJECTIVE}")

        # 创建测试参数列表（按照param_names的顺序）
        test_params_dict = {
            'd_model': 64,
            'patch_len': 8,
            'patch_stride': 4,
            'price_seq_len': 60,
            'wavelet_name': 'db4',
            'wavelet_level': 2,
            'tfactor': 2,
            'dfactor': 2,
            'mlp_hidden_dim_1': 512,
            'mlp_hidden_dim_2': 256,
            'batch_size': 16,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'dropout': 0.1,
            'epochs': 2  # 很少的轮数用于测试
        }

        # 转换为参数列表
        test_params_list = [test_params_dict[name] for name in param_names]

        print(f"   测试参数: {test_params_dict}")

        # 调用目标函数
        result = objective(test_params_list)

        print(f"✅ 目标函数测试成功")
        print(f"   优化评分: {result}")
        print(f"   类型: {type(result)}")

        if isinstance(result, (int, float)) and not np.isnan(result) and not np.isinf(result):
            print(f"   ✅ 返回值有效")
            return True
        else:
            print(f"   ❌ 返回值无效")
            return False

    except Exception as e:
        print(f"❌ 目标函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scoring_function():
    """测试评分函数"""
    print("\n📊 测试评分函数...")

    try:
        from scripts.training.bayesian_optimize_wpmixer import calculate_optimization_score

        # 创建模拟的评估指标
        test_metrics = {
            'loss': 0.5,      # MSE损失
            'mae': 0.3,       # MAE
            'r2': 0.8,        # R²
            'mape': 15.0      # MAPE
        }

        print(f"   测试指标: {test_metrics}")

        # 测试不同的优化目标
        objectives = ['mse_only', 'composite', 'mae_focused', 'r2_focused']

        for obj_type in objectives:
            score, details = calculate_optimization_score(test_metrics, obj_type)
            print(f"   {obj_type}: 评分={score:.6f}, 类型={details['score_type']}")

        print(f"✅ 评分函数测试成功")
        return True

    except Exception as e:
        print(f"❌ 评分函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_mini_optimization():
    """运行迷你优化测试"""
    print("\n🚀 运行迷你贝叶斯优化...")
    
    try:
        from skopt import gp_minimize
        from scripts.training.bayesian_optimize_wpmixer import objective, search_space
        
        print(f"   迭代次数: 3")
        print(f"   随机初始化: 2")
        
        # 运行很少次数的优化
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=3,
            n_random_starts=2,
            acq_func='EI',
            random_state=42,
            verbose=False
        )
        
        print(f"✅ 迷你优化完成")
        print(f"   最佳损失: {result.fun:.6f}")
        print(f"   评估次数: {len(result.func_vals)}")
        print(f"   损失历史: {[f'{val:.6f}' for val in result.func_vals]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 迷你优化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 贝叶斯优化测试套件")
    print("="*50)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("导入测试", test_imports),
        ("数据可用性测试", test_data_availability),
        ("模型创建测试", test_model_creation),
        ("数据准备测试", test_data_preparation),
        ("评分函数测试", test_scoring_function),
        ("目标函数测试", test_objective_function),
        ("迷你优化测试", run_mini_optimization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    print(f"\n{'='*50}")
    print(f"🏁 测试完成: {passed}/{total} 通过")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if passed == total:
        print(f"🎉 所有测试通过！贝叶斯优化系统准备就绪。")
        print(f"💡 现在可以运行完整的贝叶斯优化:")
        print(f"   python scripts/training/run_bayesian_optimization.py --quick_test")
    else:
        print(f"⚠️ 有 {total - passed} 个测试失败，请检查问题后再运行完整优化。")

if __name__ == '__main__':
    main()
