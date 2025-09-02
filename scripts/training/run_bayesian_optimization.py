#!/usr/bin/env python3
"""
贝叶斯优化WPMixer运行脚本

这个脚本提供了一个简单的接口来运行贝叶斯优化，
包含了一些额外的配置选项和错误处理。

使用方法:
    python scripts/training/run_bayesian_optimization.py
    
或者使用自定义参数:
    python scripts/training/run_bayesian_optimization.py --n_calls 30 --n_random_starts 5
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行WPMixer贝叶斯优化')
    
    parser.add_argument('--n_calls', type=int, default=50,
                       help='贝叶斯优化的总迭代次数 (默认: 50)')
    parser.add_argument('--n_random_starts', type=int, default=10,
                       help='随机初始化的次数 (默认: 10)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--cache_dir', type=str, default='experiments/cache/bayesian_optimization',
                       help='缓存目录 (默认: experiments/cache/bayesian_optimization)')
    parser.add_argument('--dry_run', action='store_true',
                       help='仅显示配置，不实际运行优化')
    parser.add_argument('--quick_test', action='store_true',
                       help='快速测试模式 (减少迭代次数)')
    
    return parser.parse_args()

def check_dependencies():
    """检查必要的依赖"""
    required_packages = ['scikit-optimize', 'torch', 'pandas', 'numpy', 'sklearn', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要的依赖包: {', '.join(missing_packages)}")
        print(f"请运行: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_availability():
    """检查数据文件是否存在"""
    data_path = 'scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print(f"请确保已经准备好价格数据文件")
        return False
    
    print(f"✅ 数据文件存在: {data_path}")
    return True

def setup_environment(args):
    """设置环境变量"""
    # 设置优化参数
    os.environ['BAYESIAN_N_CALLS'] = str(args.n_calls)
    os.environ['BAYESIAN_N_RANDOM_STARTS'] = str(args.n_random_starts)
    os.environ['BAYESIAN_RANDOM_SEED'] = str(args.random_seed)
    os.environ['BAYESIAN_CACHE_DIR'] = args.cache_dir
    
    # 创建缓存目录
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print(f"🔧 环境配置:")
    print(f"  迭代次数: {args.n_calls}")
    print(f"  随机初始化: {args.n_random_starts}")
    print(f"  随机种子: {args.random_seed}")
    print(f"  缓存目录: {args.cache_dir}")

def run_optimization(args):
    """运行贝叶斯优化"""
    script_path = 'scripts/training/bayesian_optimize_wpmixer.py'
    
    if not os.path.exists(script_path):
        print(f"❌ 优化脚本不存在: {script_path}")
        return False
    
    print(f"\n🚀 开始运行贝叶斯优化...")
    print(f"📝 脚本路径: {script_path}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 运行优化脚本
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        
        print(f"\n✅ 贝叶斯优化成功完成!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 贝叶斯优化失败!")
        print(f"错误代码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断了优化过程")
        return False
    except Exception as e:
        print(f"\n❌ 运行过程中出现错误: {e}")
        return False

def main():
    """主函数"""
    print("🎯 WPMixer贝叶斯优化运行器")
    print("="*50)
    
    # 解析参数
    args = parse_arguments()
    
    # 快速测试模式
    if args.quick_test:
        args.n_calls = 10
        args.n_random_starts = 3
        print("⚡ 快速测试模式已启用")
    
    # 干运行模式
    if args.dry_run:
        print("🔍 干运行模式 - 仅显示配置")
        setup_environment(args)
        print("\n✅ 配置检查完成，实际运行请移除 --dry_run 参数")
        return
    
    # 检查依赖
    print("🔍 检查依赖...")
    if not check_dependencies():
        return
    
    # 检查数据
    print("📊 检查数据...")
    if not check_data_availability():
        return
    
    # 设置环境
    setup_environment(args)
    
    # 运行优化
    success = run_optimization(args)
    
    if success:
        print(f"\n🎉 优化完成! 结果保存在: {args.cache_dir}")
        print(f"📋 查看结果文件:")
        print(f"  - optimization_summary.json: 优化摘要")
        print(f"  - best_params.json: 最佳参数")
        print(f"  - best_bayesian_wpmixer_model.pt: 最佳模型")
    else:
        print(f"\n💡 如果遇到问题，请检查:")
        print(f"  1. 数据文件是否存在且格式正确")
        print(f"  2. GPU内存是否足够")
        print(f"  3. 依赖包是否正确安装")
        print(f"  4. 查看错误日志文件")

if __name__ == '__main__':
    main()
