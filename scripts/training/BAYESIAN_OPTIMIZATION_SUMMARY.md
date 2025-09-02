# WPMixer贝叶斯优化实现总结

## 🎯 项目概述

已成功创建了一个专门用于WPMixer模型的贝叶斯优化系统，专注于加密货币价格回归任务。该系统**不包含GCN和新闻特征**，以简化优化过程并专注于核心的时间序列预测能力。

## 📁 创建的文件

### 1. 核心优化脚本
- **`scripts/training/bayesian_optimize_wpmixer.py`** - 主要的贝叶斯优化实现
- **`scripts/training/run_bayesian_optimization.py`** - 便捷的运行脚本
- **`scripts/training/test_bayesian_optimization.py`** - 测试套件

### 2. 文档文件
- **`scripts/training/BAYESIAN_OPTIMIZATION_README.md`** - 详细使用指南
- **`scripts/training/BAYESIAN_OPTIMIZATION_SUMMARY.md`** - 本总结文档

## ✅ 功能验证状态

根据测试结果，以下功能已验证正常工作：

### ✅ 已验证功能
1. **导入测试** - 所有必要的依赖包正常导入
2. **数据可用性** - 价格数据文件存在且可读取
3. **模型创建** - UnifiedWPMixer模型可正常创建和运行
4. **数据准备** - 数据集创建、划分、标准化正常
5. **目标函数** - 贝叶斯优化的目标函数正常工作
6. **类型转换** - numpy类型正确转换为Python原生类型

### 🔧 已修复的问题
1. **全局变量访问** - 修复了数据准备后的全局变量设置
2. **参数类型转换** - 解决了numpy类型导致的batch_size错误
3. **错误处理** - 添加了更好的异常处理，避免返回无穷大值
4. **函数签名** - 修复了目标函数的参数传递方式

## 🔍 超参数搜索空间

系统优化以下15个超参数：

| 类别 | 参数 | 搜索范围 | 说明 |
|------|------|----------|------|
| **WPMixer核心** | d_model | 32-256 | 模型维度 |
| | patch_len | 4-16 | 补丁长度 |
| | patch_stride | 2-8 | 补丁步长 |
| | price_seq_len | 30-120 | 价格序列长度 |
| | wavelet_name | db1/db4/db8/haar | 小波类型 |
| | wavelet_level | 1-4 | 小波分解层数 |
| | tfactor | 2-8 | Token混合器扩展因子 |
| | dfactor | 2-8 | 嵌入混合器扩展因子 |
| **MLP架构** | mlp_hidden_dim_1 | 256-2048 | 第一隐藏层维度 |
| | mlp_hidden_dim_2 | 128-1024 | 第二隐藏层维度 |
| **训练参数** | batch_size | 16-128 | 批次大小 |
| | learning_rate | 1e-5 - 1e-2 | 学习率（对数分布） |
| | weight_decay | 1e-6 - 1e-2 | 权重衰减（对数分布） |
| | dropout | 0.0-0.5 | Dropout率 |
| | epochs | 20-100 | 训练轮数 |

## 🚀 使用方法

### 快速开始
```bash
# 安装依赖
pip install scikit-optimize

# 运行测试（推荐先运行）
python scripts/training/test_bayesian_optimization.py

# 快速测试优化
python scripts/training/run_bayesian_optimization.py --quick_test

# 完整优化（50次迭代）
python scripts/training/run_bayesian_optimization.py

# 直接运行优化脚本
python scripts/training/bayesian_optimize_wpmixer.py
```

### 自定义参数
```bash
# 自定义迭代次数
python scripts/training/run_bayesian_optimization.py --n_calls 30 --n_random_starts 5

# 干运行（仅检查配置）
python scripts/training/run_bayesian_optimization.py --dry_run
```

## 📊 输出结果

优化完成后，在 `experiments/cache/bayesian_optimization/` 目录下生成：

### 主要文件
- **`optimization_summary.json`** - 优化摘要和最终测试结果
- **`best_params.json`** - 最佳超参数配置
- **`best_bayesian_wpmixer_model.pt`** - 最佳模型权重
- **`bayesian_optimization_results_YYYYMMDD_HHMMSS.json`** - 完整优化历史

### 评估指标
- **MSE Loss** - 均方误差损失（优化目标）
- **MAE** - 平均绝对误差
- **RMSE** - 均方根误差
- **R²** - 决定系数
- **MAPE** - 平均绝对百分比误差

## 🔧 技术特点

### 贝叶斯优化配置
- **采集函数**: Expected Improvement (EI)
- **高斯过程**: 默认RBF核
- **优化目标**: 验证集MSE损失最小化
- **默认设置**: 50次迭代，10次随机初始化

### 训练策略
- **早停机制**: 15轮无改善自动停止
- **学习率调度**: ReduceLROnPlateau
- **梯度裁剪**: 最大范数1.0
- **数据划分**: 时间序列安全划分（70%训练，15%验证，15%测试）
- **数据标准化**: 仅使用训练集拟合标准化器

### 模型配置
- **任务类型**: 回归（价格预测）
- **GCN**: 禁用
- **新闻特征**: 禁用
- **币种**: BTC, ETH, BNB, XRP, LTC, DOGE, SOL, AVAX

## 💡 使用建议

### 首次使用
1. 先运行测试套件确保环境正常
2. 使用快速测试模式验证流程
3. 根据需要调整搜索空间
4. 运行完整优化

### 性能优化
- 如果GPU内存不足，减少batch_size和d_model的搜索范围
- 如果优化收敛慢，增加n_random_starts
- 如果训练不稳定，检查学习率范围

### 结果分析
- 查看optimization_summary.json了解整体性能
- 分析优化历史找出参数趋势
- 使用最佳参数进行进一步实验

## 🎉 总结

贝叶斯优化系统已成功实现并通过测试验证。主要优势：

1. **自动化超参数调优** - 无需手动尝试参数组合
2. **高效搜索** - 贝叶斯优化比网格搜索更高效
3. **完整流程** - 从数据准备到模型评估的端到端流程
4. **易于使用** - 提供多种运行方式和详细文档
5. **可扩展性** - 易于修改搜索空间和优化目标

系统现在已准备好用于WPMixer模型的超参数优化，专注于加密货币价格回归任务。

## 🎯 多目标优化功能（新增）

### 优化目标选项

系统现在支持多种优化目标，不再仅限于MSE损失：

1. **`composite`** - 综合优化（推荐）
   - 同时优化MSE、MAE、R²和MAPE
   - 默认权重：MSE(40%) + MAE(30%) + R²(20%) + MAPE(10%)
   - 提供最全面的模型评估

2. **`mse_only`** - 仅优化MSE损失
   - 传统的单一目标优化
   - 适合快速验证和基础优化

3. **`mae_focused`** - 主要优化MAE
   - MAE(80%) + R²惩罚(20%)
   - 适合关注绝对误差的场景

4. **`r2_focused`** - 主要优化R²
   - R²惩罚(70%) + MSE(30%)
   - 适合关注模型解释能力的场景

### 配置方法

1. **使用配置文件**（推荐）：
```python
# 修改 bayesian_optimization_config.py
OPTIMIZATION_OBJECTIVE = 'composite'
COMPOSITE_WEIGHTS = {
    'mse_weight': 0.4,
    'mae_weight': 0.3,
    'r2_weight': 0.2,
    'mape_weight': 0.1
}
```

2. **使用预定义策略**：
```python
from bayesian_optimization_config import apply_strategy
apply_strategy('accuracy_focused')      # 专注准确性
apply_strategy('interpretability_focused')  # 专注解释能力
apply_strategy('robustness_focused')    # 专注鲁棒性
```

### 评分计算

综合评分公式：
```
score = w1×(MSE/100) + w2×(MAE/10) + w3×max(0,1-R²) + w4×min(MAPE/100,1)
```

其中：
- MSE、MAE、MAPE越小越好
- R²越大越好（使用1-R²作为惩罚项）
- 归一化处理避免不同指标量级差异

### 预定义策略

- **balanced**: 平衡的综合优化
- **accuracy_focused**: 专注预测准确性
- **interpretability_focused**: 专注模型解释能力
- **robustness_focused**: 专注模型鲁棒性
- **simple_mse**: 仅优化MSE
- **mae_priority**: 主要优化MAE
- **r2_priority**: 主要优化R²

## 💾 最佳参数保存功能（新增）

### 多格式参数保存

系统现在会自动将最佳参数保存为多种格式：

1. **JSON格式** (`best_params.json`)
   - 标准的参数字典格式
   - 便于程序读取和解析

2. **Python配置文件** (`best_params_YYYYMMDD_HHMMSS.py`)
   - 可直接导入的Python模块
   - 包含配置类和使用示例
   - 包含优化结果和测试指标

3. **YAML格式** (`best_params_YYYYMMDD_HHMMSS.yaml`)
   - 人类可读的配置格式
   - 结构化的参数组织
   - 便于版本控制和文档化

### 参数加载工具

提供了专门的工具脚本来加载和使用最佳参数：

#### `load_best_params.py` - 参数加载工具
```bash
# 查看最佳参数摘要
python scripts/training/load_best_params.py

# 加载不同格式
python scripts/training/load_best_params.py --format yaml
python scripts/training/load_best_params.py --format python

# 创建配置对象
python scripts/training/load_best_params.py --create-config
```

#### `use_best_params_example.py` - 使用示例
```bash
# 运行完整的使用示例
python scripts/training/use_best_params_example.py
```

### 参数文件内容

每个参数文件都包含：

- **WPMixer核心参数**: d_model, patch_len, wavelet_name等
- **MLP架构参数**: hidden_dim_1, hidden_dim_2
- **训练参数**: batch_size, learning_rate, weight_decay等
- **优化结果**: 最佳评分、优化目标类型
- **测试指标**: MSE, MAE, R², MAPE等性能指标

### 使用方法

1. **程序化加载**:
```python
from scripts.training.load_best_params import load_best_params_json
params = load_best_params_json('path/to/best_params.json')
```

2. **直接导入Python配置**:
```python
from experiments.cache.bayesian_optimization.best_params_20240101_120000 import *
config = get_wpmixer_config()
```

3. **配置对象创建**:
```python
wpmixer_config = create_wpmixer_config_from_params(params)
training_config = create_training_config_from_params(params)
```
