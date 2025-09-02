# 统一训练脚本目录

本目录包含了所有主要模型的统一训练脚本，使用相同的图构建方法和配置结构，便于公平比较。

## 📁 文件结构

```
unified_scripts/
├── train_timemixer.py    # TimeMixer模型训练脚本
├── train_timexer.py      # TimeXer模型训练脚本  
├── train_cnn.py          # CNN模型训练脚本
├── train_lstm.py         # LSTM模型训练脚本
├── run_all_models.py     # 批量运行所有模型的脚本
└── README.md             # 本说明文件
```

## 🎯 主要改进

### 1. 统一的图构建方法
- **所有脚本都使用相同的图构建配置**
- **基于实验结果，默认使用`original`方法**（简单相关性图，阈值0.6）
- **支持边权重**的GCN实现

### 2. 统一的结果保存方式
- **模型文件命名**：`{ModelName}_{TaskType}_{GCN}_{News}_{model_file}.pt`
  - 例如：`TimeMixer_regression_with_gcn_with_news_best_timemixer_model.pt`
- **预测结果命名**：`test_predictions_{ModelName}_{TaskType}_{GCN}_{News}.csv`
  - 例如：`test_predictions_TimeMixer_regression_with_gcn_with_news.csv`

### 3. 统一的配置结构
所有脚本都包含相同的配置部分：
- **Master Switches**: 任务类型、GCN开关、新闻特征开关
- **Graph Construction**: 图构建方法和参数
- **Model Parameters**: 各模型特定的参数
- **Training Parameters**: 训练超参数

## 🚀 使用方法

### 单独运行模型
```bash
# 运行TimeMixer模型
python scripts/training/unified_scripts/train_timemixer.py

# 运行TimeXer模型  
python scripts/training/unified_scripts/train_timexer.py

# 运行CNN模型
python scripts/training/unified_scripts/train_cnn.py

# 运行LSTM模型
python scripts/training/unified_scripts/train_lstm.py
```

### 批量运行所有模型
```bash
# 运行所有模型并自动比较结果
python scripts/training/unified_scripts/run_all_models.py
```

## ⚙️ 配置说明

### 图构建配置
```python
# 基于实验结果，原始方法表现最佳
GRAPH_METHOD = 'original'  # 推荐设置

# 可选方法：
# - 'original': 简单相关性图（推荐）
# - 'multi_layer': 多层图结构
# - 'dynamic': 动态时变图
# - 'domain_knowledge': 领域知识图
# - 'attention_based': 注意力图
```

### 模型开关
```python
USE_GCN = True           # 启用图卷积网络
USE_NEWS_FEATURES = True # 启用新闻特征
TASK_TYPE = 'regression' # 回归任务
```

### 训练参数
```python
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
```

## 📊 结果分析

### 自动生成的文件
1. **模型文件**: `experiments/cache/{model_variant}_{model_name}.pt`
2. **预测结果**: `experiments/cache/test_predictions/test_predictions_{model_variant}.csv`
3. **统计信息**: `experiments/cache/test_predictions/test_statistics_{model_variant}.csv`
4. **比较结果**: `experiments/cache/unified_comparison/`

### 关键指标
- **MAE**: 平均绝对误差（越小越好）
- **New MAE**: 总和比值MAE（越接近1.0越好）
- **R²**: 决定系数（越大越好）
- **MAPE**: 平均绝对百分比误差（越小越好）

## 🔧 自定义配置

### 修改图构建方法
在脚本中修改：
```python
GRAPH_METHOD = 'domain_knowledge'  # 改为其他方法
```

### 调整模型参数
每个脚本都有特定的模型参数部分，例如TimeMixer：
```python
# --- TimeMixer Model Parameters ---
NEWS_PROCESSED_DIM = 64
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 128
```

### 修改数据集参数
```python
PRICE_SEQ_LEN = 90      # 序列长度
NORM_TYPE = 'minmax'    # 归一化方式
COIN_NAMES = [...]      # 币种列表
```

## 📈 实验结果总结

基于图构建方法的比较实验：

| 方法 | MAE | New MAE | R² | 特点 |
|------|-----|---------|----|----|
| **Original** | **0.312** | **0.710** | **0.825** | 🏆 最佳性能 |
| Domain Knowledge | 0.442 | 2.425 | 0.670 | 过度连接 |
| Dynamic | 0.477 | 1.235 | 0.602 | 中等性能 |

**结论**: 简单的相关性图方法表现最佳，因此所有统一脚本都默认使用此方法。

## 🛠️ 故障排除

### 常见问题
1. **CUDA内存不足**: 减小`BATCH_SIZE`
2. **模型文件冲突**: 检查`CACHE_DIR`路径
3. **数据路径错误**: 确认`PRICE_CSV_PATH`和`NEWS_FEATURES_FOLDER`

### 调试模式
在脚本中设置：
```python
EPOCHS = 3          # 快速测试
BATCH_SIZE = 16     # 减小批次大小
```

## 📝 更新日志

- **2025-07-15**: 创建统一训练脚本目录
- **图构建优化**: 基于实验结果选择最佳方法
- **结果保存优化**: 使用模型名称而非时间戳命名
- **批量比较**: 添加自动化模型比较功能

---

**注意**: 运行前请确保所有依赖已安装，数据文件路径正确。
