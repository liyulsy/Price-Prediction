# WPMixer贝叶斯优化指南

这个文档介绍如何使用贝叶斯优化来自动调优WPMixer模型的超参数，专门用于加密货币价格回归任务。

## 📋 概述

贝叶斯优化是一种高效的超参数调优方法，特别适合于：
- 目标函数评估成本高（如深度学习模型训练）
- 参数空间复杂且高维
- 需要在有限的计算资源下找到最优解

本实现专注于WPMixer模型的价格回归任务，**不包含GCN和新闻特征**，以简化优化过程。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install scikit-optimize torch pandas numpy scikit-learn tqdm
```

### 2. 准备数据

确保价格数据文件存在：
```
scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv
```

### 3. 运行优化

#### 基本用法
```bash
python scripts/training/run_bayesian_optimization.py
```

#### 自定义参数
```bash
# 快速测试（10次迭代）
python scripts/training/run_bayesian_optimization.py --quick_test

# 自定义迭代次数
python scripts/training/run_bayesian_optimization.py --n_calls 30 --n_random_starts 5

# 干运行（仅检查配置）
python scripts/training/run_bayesian_optimization.py --dry_run
```

#### 直接运行优化脚本
```bash
python scripts/training/bayesian_optimize_wpmixer.py
```

## 🔧 配置说明

### 优化目标配置

系统支持多种优化目标，可以通过修改配置文件或直接修改脚本来选择：

#### 优化目标类型

1. **`mse_only`** - 仅优化MSE损失
   - 最简单直接的方法
   - 适合快速验证和基础优化

2. **`composite`** - 综合优化多个指标（推荐）
   - 同时考虑MSE、MAE、R²和MAPE
   - 提供最全面的模型评估
   - 可自定义各指标权重

3. **`mae_focused`** - 主要优化MAE
   - 80%权重给MAE，20%给R²
   - 适合关注绝对误差的场景

4. **`r2_focused`** - 主要优化R²
   - 70%权重给R²，30%给MSE
   - 适合关注模型解释能力的场景

#### 综合评分权重（composite模式）

默认权重配置：
- **MSE损失**: 40% - 基础损失函数
- **MAE**: 30% - 平均绝对误差，更直观
- **R²**: 20% - 模型解释能力
- **MAPE**: 10% - 相对误差百分比

#### 配置方法

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
apply_strategy('accuracy_focused')  # 专注准确性
apply_strategy('interpretability_focused')  # 专注解释能力
```

3. **直接修改脚本**：
```python
# 在 bayesian_optimize_wpmixer.py 中修改
OPTIMIZATION_OBJECTIVE = 'mae_focused'
```

### 超参数搜索空间

| 参数 | 范围 | 说明 |
|------|------|------|
| `d_model` | 32-256 | 模型维度 |
| `patch_len` | 4-16 | 补丁长度 |
| `patch_stride` | 2-8 | 补丁步长 |
| `price_seq_len` | 30-120 | 价格序列长度 |
| `wavelet_name` | db1/db4/db8/haar | 小波类型 |
| `wavelet_level` | 1-4 | 小波分解层数 |
| `tfactor` | 2-8 | Token混合器扩展因子 |
| `dfactor` | 2-8 | 嵌入混合器扩展因子 |
| `mlp_hidden_dim_1` | 256-2048 | MLP第一隐藏层维度 |
| `mlp_hidden_dim_2` | 128-1024 | MLP第二隐藏层维度 |
| `batch_size` | 16-128 | 批次大小 |
| `learning_rate` | 1e-5 - 1e-2 | 学习率（对数均匀分布） |
| `weight_decay` | 1e-6 - 1e-2 | 权重衰减（对数均匀分布） |
| `dropout` | 0.0-0.5 | Dropout率 |
| `epochs` | 20-100 | 训练轮数 |

### 固定配置

- **任务类型**: 回归（价格预测）
- **GCN**: 禁用
- **新闻特征**: 禁用
- **数据归一化**: 标准化
- **早停耐心值**: 15轮
- **验证集比例**: 15%
- **测试集比例**: 15%

## 📊 输出文件

优化完成后，会在 `experiments/cache/bayesian_optimization/` 目录下生成：

### 主要文件
- `optimization_summary.json`: 优化摘要
- `best_bayesian_wpmixer_model.pt`: 最佳模型权重
- `bayesian_optimization_results_YYYYMMDD_HHMMSS.json`: 详细优化历史

### 最佳参数文件（多种格式）
- `best_params.json`: 最佳参数配置（JSON格式）
- `best_params_YYYYMMDD_HHMMSS.py`: Python配置文件格式
- `best_params_YYYYMMDD_HHMMSS.yaml`: YAML配置文件格式

### 文件内容说明

#### `optimization_summary.json`
```json
{
  "optimization_completed": true,
  "best_validation_loss": 0.001234,
  "best_params": {...},
  "test_metrics": {...},
  "optimization_time_seconds": 3600,
  "n_calls": 50,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `best_params.json`
```json
{
  "d_model": 128,
  "patch_len": 8,
  "patch_stride": 4,
  "price_seq_len": 60,
  "wavelet_name": "db4",
  "wavelet_level": 2,
  "tfactor": 4,
  "dfactor": 4,
  "mlp_hidden_dim_1": 1024,
  "mlp_hidden_dim_2": 512,
  "batch_size": 64,
  "learning_rate": 0.0001,
  "weight_decay": 0.0001,
  "dropout": 0.1,
  "epochs": 50
}
```

## 🎯 优化策略

### 贝叶斯优化配置
- **采集函数**: Expected Improvement (EI)
- **高斯过程**: 默认RBF核
- **优化目标**: 验证集损失最小化

### 训练策略
- **早停机制**: 15轮无改善自动停止
- **学习率调度**: ReduceLROnPlateau
- **梯度裁剪**: 最大范数1.0
- **数据划分**: 时间序列安全划分（避免数据泄露）

## 📈 监控和调试

### 实时监控
优化过程中会显示：
- 当前评估的参数组合
- 每轮的训练和验证损失
- 早停信息
- 最佳参数更新

### 性能指标
最终评估包括：
- **损失**: MSE损失
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数
- **MAPE**: 平均绝对百分比误差

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 减少 `batch_size` 搜索范围
   - 减少 `d_model` 最大值
   - 使用更小的 `price_seq_len`

2. **优化收敛慢**
   - 增加 `n_random_starts`
   - 调整搜索空间范围
   - 检查数据质量

3. **训练不稳定**
   - 检查学习率范围
   - 增加早停耐心值
   - 检查梯度裁剪设置

### 调试技巧

1. **使用快速测试模式**
   ```bash
   python scripts/training/run_bayesian_optimization.py --quick_test
   ```

2. **检查单个参数组合**
   修改 `bayesian_optimize_wpmixer.py` 中的固定参数进行测试

3. **监控GPU使用**
   ```bash
   nvidia-smi -l 1
   ```

## 🎯 使用最佳参数

优化完成后，可以通过多种方式使用最佳参数：

### 1. 加载参数工具

```bash
# 查看最佳参数摘要
python scripts/training/load_best_params.py

# 加载YAML格式参数
python scripts/training/load_best_params.py --format yaml

# 加载指定文件
python scripts/training/load_best_params.py --file path/to/best_params.json

# 创建配置对象示例
python scripts/training/load_best_params.py --create-config
```

### 2. 在代码中使用

```python
from scripts.training.load_best_params import (
    load_best_params_json, create_wpmixer_config_from_params,
    create_training_config_from_params
)

# 加载最佳参数
params = load_best_params_json('experiments/cache/bayesian_optimization/best_params.json')

# 创建模型配置
wpmixer_config = create_wpmixer_config_from_params(params)

# 创建训练配置
training_config = create_training_config_from_params(params)

# 创建模型
model = UnifiedWPMixer(
    configs=wpmixer_config,
    use_gcn=False,
    mlp_hidden_dim_1=training_config['mlp_hidden_dim_1'],
    mlp_hidden_dim_2=training_config['mlp_hidden_dim_2'],
    num_classes=1
)
```

### 3. 直接导入Python配置

```python
# 导入生成的Python配置文件
from experiments.cache.bayesian_optimization.best_params_20240101_120000 import *

# 使用配置
config = get_wpmixer_config()
training_config = get_training_config()
```

### 4. 使用示例脚本

```bash
# 运行完整的使用示例
python scripts/training/use_best_params_example.py
```

## 📚 扩展使用

### 自定义搜索空间
修改 `bayesian_optimize_wpmixer.py` 中的 `search_space` 列表：

```python
search_space = [
    Integer(64, 512, name='d_model'),  # 扩大搜索范围
    # 添加新参数...
]
```

### 自定义优化目标
修改 `objective` 函数的返回值：

```python
# 使用R²作为优化目标（最大化）
return -val_metrics['r2']  # 负号因为优化器最小化目标
```

### 多目标优化
可以结合多个指标：

```python
# 综合考虑损失和R²
combined_score = val_loss - 0.1 * val_metrics['r2']
return combined_score
```

## 📞 支持

如果遇到问题，请检查：
1. 错误日志文件
2. GPU内存使用情况
3. 数据文件完整性
4. 依赖包版本兼容性
