"""
贝叶斯优化配置文件

这个文件包含了贝叶斯优化的各种配置选项，
用户可以通过修改这个文件来自定义优化行为。
"""

# =============================================================================
# 优化目标配置
# =============================================================================

# 优化目标选择：
# 'mse_only': 仅优化MSE损失 - 最简单直接的方法
# 'composite': 综合优化MSE、MAE、R²和MAPE - 推荐用于全面评估
# 'mae_focused': 主要优化MAE，辅助考虑R² - 适合关注绝对误差
# 'r2_focused': 主要优化R²，辅助考虑MSE - 适合关注模型解释能力
OPTIMIZATION_OBJECTIVE = 'composite'

# =============================================================================
# 综合评分权重配置（仅在OPTIMIZATION_OBJECTIVE='composite'时使用）
# =============================================================================

COMPOSITE_WEIGHTS = {
    'mse_weight': 0.4,      # MSE损失权重 - 基础损失函数
    'mae_weight': 0.3,      # MAE权重 - 平均绝对误差，更直观
    'r2_weight': 0.2,       # R²权重 - 模型解释能力
    'mape_weight': 0.1      # MAPE权重 - 相对误差百分比
}

# 注意：所有权重之和应该等于1.0
assert abs(sum(COMPOSITE_WEIGHTS.values()) - 1.0) < 1e-6, "权重之和必须等于1.0"

# =============================================================================
# 归一化参数配置
# =============================================================================

# 用于将不同指标缩放到相似范围，避免某个指标主导优化过程
NORMALIZATION_PARAMS = {
    'mse_scale': 100.0,     # MSE损失通常在0-100范围
    'mae_scale': 10.0,      # MAE通常在0-10范围  
    'mape_scale': 100.0     # MAPE是百分比，通常在0-100范围
}

# =============================================================================
# 贝叶斯优化参数
# =============================================================================

N_CALLS = 50                # 贝叶斯优化的总迭代次数
N_RANDOM_STARTS = 10        # 随机初始化的次数（探索阶段）

# =============================================================================
# 训练参数
# =============================================================================

EARLY_STOPPING_PATIENCE = 15   # 早停耐心值
MIN_DELTA = 1e-6               # 最小改善阈值
VALIDATION_SPLIT_RATIO = 0.15  # 验证集比例
TEST_SPLIT_RATIO = 0.15        # 测试集比例

# =============================================================================
# 预定义的优化策略
# =============================================================================

OPTIMIZATION_STRATEGIES = {
    'balanced': {
        'objective': 'composite',
        'weights': {'mse_weight': 0.4, 'mae_weight': 0.3, 'r2_weight': 0.2, 'mape_weight': 0.1},
        'description': '平衡的综合优化策略'
    },
    
    'accuracy_focused': {
        'objective': 'composite', 
        'weights': {'mse_weight': 0.5, 'mae_weight': 0.4, 'r2_weight': 0.1, 'mape_weight': 0.0},
        'description': '专注于预测准确性'
    },
    
    'interpretability_focused': {
        'objective': 'composite',
        'weights': {'mse_weight': 0.3, 'mae_weight': 0.2, 'r2_weight': 0.4, 'mape_weight': 0.1},
        'description': '专注于模型解释能力'
    },
    
    'robustness_focused': {
        'objective': 'composite',
        'weights': {'mse_weight': 0.2, 'mae_weight': 0.5, 'r2_weight': 0.2, 'mape_weight': 0.1},
        'description': '专注于模型鲁棒性'
    },
    
    'simple_mse': {
        'objective': 'mse_only',
        'weights': None,
        'description': '仅优化MSE损失'
    },
    
    'mae_priority': {
        'objective': 'mae_focused',
        'weights': None,
        'description': '主要优化MAE'
    },
    
    'r2_priority': {
        'objective': 'r2_focused', 
        'weights': None,
        'description': '主要优化R²'
    }
}

# =============================================================================
# 快速配置函数
# =============================================================================

def apply_strategy(strategy_name):
    """
    应用预定义的优化策略
    
    Args:
        strategy_name: 策略名称，可选值见OPTIMIZATION_STRATEGIES
    """
    global OPTIMIZATION_OBJECTIVE, COMPOSITE_WEIGHTS
    
    if strategy_name not in OPTIMIZATION_STRATEGIES:
        available = list(OPTIMIZATION_STRATEGIES.keys())
        raise ValueError(f"未知策略: {strategy_name}. 可用策略: {available}")
    
    strategy = OPTIMIZATION_STRATEGIES[strategy_name]
    OPTIMIZATION_OBJECTIVE = strategy['objective']
    
    if strategy['weights'] is not None:
        COMPOSITE_WEIGHTS.update(strategy['weights'])
    
    print(f"✅ 已应用优化策略: {strategy_name}")
    print(f"   描述: {strategy['description']}")
    print(f"   优化目标: {OPTIMIZATION_OBJECTIVE}")
    if strategy['weights'] is not None:
        print(f"   权重配置: {strategy['weights']}")

def get_current_config():
    """获取当前配置的摘要"""
    config = {
        'optimization_objective': OPTIMIZATION_OBJECTIVE,
        'composite_weights': COMPOSITE_WEIGHTS.copy() if OPTIMIZATION_OBJECTIVE == 'composite' else None,
        'normalization_params': NORMALIZATION_PARAMS.copy(),
        'n_calls': N_CALLS,
        'n_random_starts': N_RANDOM_STARTS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE
    }
    return config

def print_current_config():
    """打印当前配置"""
    print("🔧 当前贝叶斯优化配置:")
    print(f"  优化目标: {OPTIMIZATION_OBJECTIVE}")
    
    if OPTIMIZATION_OBJECTIVE == 'composite':
        print(f"  综合权重:")
        for key, value in COMPOSITE_WEIGHTS.items():
            print(f"    {key}: {value:.1%}")
    
    print(f"  迭代次数: {N_CALLS}")
    print(f"  随机初始化: {N_RANDOM_STARTS}")
    print(f"  早停耐心值: {EARLY_STOPPING_PATIENCE}")

# =============================================================================
# 使用示例
# =============================================================================

if __name__ == '__main__':
    print("🎯 贝叶斯优化配置示例")
    print("="*50)
    
    # 显示当前配置
    print_current_config()
    
    print("\n📋 可用的优化策略:")
    for name, strategy in OPTIMIZATION_STRATEGIES.items():
        print(f"  {name}: {strategy['description']}")
    
    print("\n💡 使用方法:")
    print("1. 直接修改此文件中的配置变量")
    print("2. 或者在代码中调用 apply_strategy('strategy_name')")
    print("3. 然后运行贝叶斯优化脚本")
    
    print("\n🔧 配置示例:")
    print("# 应用专注于准确性的策略")
    print("apply_strategy('accuracy_focused')")
    print()
    print("# 或者自定义权重")
    print("COMPOSITE_WEIGHTS['mae_weight'] = 0.5")
    print("COMPOSITE_WEIGHTS['mse_weight'] = 0.3")
