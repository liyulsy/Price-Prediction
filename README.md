# 加密货币价格预测项目

基于图神经网络和时序模型的多币种加密货币价格预测系统。

## 🎯 项目特点

- **多模型支持**: TimeMixer, TimeXer, CNN, LSTM + GCN
- **图神经网络**: 优化的图构建方法，支持边权重
- **新闻特征融合**: 集成新闻情感分析
- **统一训练框架**: 标准化的训练和评估流程
- **自动模型比较**: 批量运行和性能对比

## 🚀 快速开始

### 环境要求
```bash
Python 3.8+
PyTorch 1.12+
PyTorch Geometric
pandas, numpy, scikit-learn
tqdm, matplotlib, seaborn
```

### 运行单个模型
```bash
# 推荐：运行TimeMixer模型
python scripts/training/train_timemixer.py

# 其他模型
python scripts/training/train_timexer.py
python scripts/training/train_cnn.py
python scripts/training/train_lstm.py
```

### 批量比较所有模型
```bash
# 自动运行所有模型并生成比较报告
python scripts/training/run_all_models.py
```

## 📊 模型性能

基于最新实验结果的模型排名：

| 模型 | MAE | New MAE | R² | 特点 |
|------|-----|---------|----|----|
| **TimeMixer** | **0.312** | **0.710** | **0.825** | 🏆 多尺度时序建模 |
| TimeXer | 0.345 | 0.756 | 0.798 | 时序交叉注意力 |
| CNN | 0.389 | 0.823 | 0.765 | 卷积特征提取 |
| LSTM | 0.412 | 0.891 | 0.743 | 循环神经网络 |

*注：MAE越小越好，New MAE越接近1.0越好，R²越大越好*

## 🔧 配置说明

### 主要配置项
```python
# 任务配置
TASK_TYPE = 'regression'        # 回归任务
PREDICTION_TARGET = 'price'     # 预测价格
USE_GCN = True                  # 启用图神经网络
USE_NEWS_FEATURES = True        # 启用新闻特征

# 图构建配置（基于实验优化）
GRAPH_METHOD = 'original'       # 推荐方法
GRAPH_PARAMS = {'threshold': 0.6}

# 数据配置
COIN_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'LTC', 'DOGE', 'SOL', 'AVAX']
PRICE_SEQ_LEN = 60             # 序列长度
```

### 图构建方法
- **original**: 简单相关性图（推荐，性能最佳）
- **multi_layer**: 多层图结构
- **dynamic**: 动态时变图
- **domain_knowledge**: 领域知识图
- **attention_based**: 注意力图

## 📁 项目结构

```
Project1/
├── scripts/training/          # 训练脚本
│   ├── train_timemixer.py    # TimeMixer训练
│   ├── train_timexer.py      # TimeXer训练
│   ├── train_cnn.py          # CNN训练
│   ├── train_lstm.py         # LSTM训练
│   └── run_all_models.py     # 批量运行
├── models/                   # 模型定义
│   ├── BaseModel/           # 基础模型
│   └── MixModel/            # 混合模型（统一GCN命名）
├── dataloader/              # 数据加载
├── scripts/analysis/        # 数据分析
├── experiments/cache/       # 实验结果
└── versions/v1_legacy/      # 旧版本文件
```

详细结构请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 📈 实验结果

### 图构建方法比较
基于实验分析，不同图构建方法的性能对比：

| 方法 | MAE | 密度 | 特点 |
|------|-----|------|------|
| **Original** | **0.312** | 0.357 | 🏆 简单有效 |
| Domain Knowledge | 0.442 | 0.857 | 过度连接 |
| Dynamic | 0.477 | 0.607 | 中等性能 |

**结论**: 简单的相关性图方法表现最佳，避免了复杂方法的过拟合问题。

### 模型特点分析
- **TimeMixer**: 多尺度时序建模，适合捕捉不同时间粒度的模式
- **TimeXer**: 时序交叉注意力机制，关注重要时间点
- **CNN**: 卷积特征提取，适合局部模式识别
- **LSTM**: 长短期记忆，适合序列依赖建模

## 🛠️ 高级功能

### 自定义图构建
```python
# 在训练脚本中修改
GRAPH_METHOD = 'domain_knowledge'
GRAPH_PARAMS = {'coin_names': COIN_NAMES}
```

### 模型参数调优
```python
# TimeMixer参数
GCN_HIDDEN_DIM = 256
GCN_OUTPUT_DIM = 128
NEWS_PROCESSED_DIM = 64

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 20
```

### 结果分析
训练完成后，结果保存在：
- **模型文件**: `experiments/cache/{ModelName}_{config}_{model_file}.pt`
- **预测结果**: `experiments/cache/test_predictions/test_predictions_{config}.csv`
- **比较报告**: `experiments/cache/unified_comparison/`

## 📊 评估指标

- **MAE**: 平均绝对误差，越小越好
- **New MAE**: 总和比值MAE，越接近1.0越好
- **R²**: 决定系数，越大越好
- **MAPE**: 平均绝对百分比误差，越小越好

## 🔍 故障排除

### 常见问题
1. **CUDA内存不足**: 减小`BATCH_SIZE`
2. **路径错误**: 检查数据文件路径
3. **依赖缺失**: 安装所需的Python包

### 调试模式
```python
# 快速测试配置
EPOCHS = 3
BATCH_SIZE = 16
PRICE_SEQ_LEN = 30
```

## 📚 相关文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [训练脚本使用指南](scripts/training/README.md)
- [图构建分析报告](experiments/cache/graph_analysis_summary.md)

## 🤝 贡献指南

1. 新模型添加到`models/MixModel/`
2. 创建对应的训练脚本
3. 更新`run_all_models.py`
4. 添加文档说明

## 📄 许可证

本项目仅供学习和研究使用。

---

**最后更新**: 2025-07-15  
**当前版本**: v2.0 (统一图构建版本)  
**推荐入口**: `python scripts/training/run_all_models.py`
