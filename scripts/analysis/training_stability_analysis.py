#!/usr/bin/env python3
"""
训练稳定性分析脚本
分析训练结果的变异性来源和改进建议
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

def analyze_training_variability():
    """分析训练变异性的可能原因"""
    
    print("🔍 训练结果变异性分析")
    print("="*60)
    
    print("\n📋 可能的变异性来源:")
    print("1. 随机性来源:")
    print("   - 权重初始化随机性")
    print("   - 数据加载顺序随机性 (DataLoader shuffle)")
    print("   - Dropout随机性")
    print("   - 批次采样随机性")
    print("   - CUDA操作的非确定性")
    
    print("\n2. 数据相关:")
    print("   - 训练/验证/测试集划分随机性")
    print("   - 时间序列数据的时间敏感性")
    print("   - 新闻数据的噪声和稀疏性")
    
    print("\n3. 模型相关:")
    print("   - 学习率调度器的敏感性")
    print("   - 早停机制的随机性")
    print("   - GCN图结构的敏感性")
    
    print("\n4. 训练相关:")
    print("   - 训练轮数不足")
    print("   - 局部最优解")
    print("   - 梯度消失/爆炸")

def suggest_improvements():
    """提供改进建议"""
    
    print("\n💡 改进建议:")
    print("="*60)
    
    print("\n🎯 提高结果稳定性:")
    print("1. 完全固定随机性:")
    print("   ✅ 已实现: set_random_seeds() 函数")
    print("   ✅ 已实现: 固定数据集划分种子")
    print("   ✅ 已实现: CUDA确定性设置")
    
    print("\n2. 增加训练稳定性:")
    print("   ✅ 已实现: 增加训练轮数到50")
    print("   ✅ 已实现: 早停机制 (patience=10)")
    print("   🔄 建议: 使用学习率预热")
    print("   🔄 建议: 梯度裁剪")
    
    print("\n3. 多次运行统计:")
    print("   ✅ 已实现: 批量实验脚本")
    print("   🔄 建议: 每个配置至少运行10次")
    print("   🔄 建议: 使用统计显著性测试")
    
    print("\n4. 数据处理改进:")
    print("   🔄 建议: 使用固定的时间窗口划分")
    print("   🔄 建议: 新闻特征降噪")
    print("   🔄 建议: 数据增强技术")

def create_training_config_recommendations():
    """创建训练配置建议"""
    
    recommendations = {
        "random_seed_control": {
            "description": "完全控制随机性",
            "implemented": True,
            "code": """
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
            """
        },
        
        "training_stability": {
            "description": "训练稳定性改进",
            "implemented": "partial",
            "suggestions": [
                "增加训练轮数 (50-100)",
                "使用学习率预热",
                "添加梯度裁剪",
                "使用更稳定的优化器 (AdamW)",
                "调整学习率调度策略"
            ]
        },
        
        "data_consistency": {
            "description": "数据一致性",
            "implemented": "partial", 
            "suggestions": [
                "使用固定的时间窗口划分而非随机划分",
                "对新闻特征进行更好的预处理",
                "使用更稳定的归一化方法",
                "考虑数据泄漏问题"
            ]
        },
        
        "evaluation_robustness": {
            "description": "评估鲁棒性",
            "implemented": True,
            "suggestions": [
                "多次运行取平均",
                "使用置信区间",
                "统计显著性测试",
                "交叉验证"
            ]
        }
    }
    
    return recommendations

def generate_improved_training_script():
    """生成改进的训练脚本建议"""
    
    print("\n🔧 改进的训练配置建议:")
    print("="*60)
    
    improved_config = """
# 改进的训练参数
BATCH_SIZE = 32
EPOCHS = 100  # 增加训练轮数
LEARNING_RATE = 0.001  # 稍微增加学习率
WEIGHT_DECAY = 1e-4  # 增加正则化
WARMUP_EPOCHS = 10  # 学习率预热
GRADIENT_CLIP_NORM = 1.0  # 梯度裁剪
EARLY_STOPPING_PATIENCE = 15  # 增加早停耐心
MIN_DELTA = 1e-5  # 更严格的改进阈值

# 使用更稳定的优化器
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8
)

# 学习率调度器改进
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=20, 
    T_mult=2, 
    eta_min=1e-6
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    """
    
    print(improved_config)

def main():
    """主函数"""
    print("🔬 训练稳定性分析和改进建议")
    print("="*80)
    
    # 分析变异性来源
    analyze_training_variability()
    
    # 提供改进建议
    suggest_improvements()
    
    # 生成配置建议
    recommendations = create_training_config_recommendations()
    
    # 生成改进的训练脚本
    generate_improved_training_script()
    
    print("\n📊 期望的实验结果模式:")
    print("="*60)
    print("如果改进措施有效，你应该看到:")
    print("1. 🎯 GCN + News > GCN Only ≈ News Only > Baseline")
    print("2. 📉 各配置的标准差显著降低")
    print("3. 📈 成功率接近100%")
    print("4. 🔄 结果在多次运行间保持一致")
    
    print("\n🚀 下一步行动:")
    print("1. 运行批量实验: python scripts/training/batch_experiment_timexer.py")
    print("2. 分析结果的统计显著性")
    print("3. 如果仍有变异性，考虑进一步的改进措施")

if __name__ == "__main__":
    main()
