# 加密货币价格预测项目

这是一个基于深度学习的加密货币价格预测项目，集成了多种机器学习模型和图神经网络。

## 项目结构

```
├── README.md                 # 项目说明文档
├── main.py                   # 主程序入口
├── config.py                 # 配置文件
├── models/                   # 模型定义
│   ├── BaseModel/           # 基础模型
│   ├── MixModel/            # 混合模型
│   └── layers/              # 自定义层
├── dataloader/              # 数据加载器
│   ├── __init__.py
│   ├── base_loader.py       # 基础数据加载器
│   └── gnn_loader.py        # 图神经网络数据加载器
├── utils/                   # 工具函数
│   └── masking.py           # 数据掩码工具
├── scripts/                 # 脚本文件
│   ├── training/            # 训练脚本
│   │   ├── train_*.py       # 各种训练脚本
│   │   ├── unified_*.py     # 统一训练脚本
│   │   └── price_prediction_unified_style.py
│   ├── optimization/        # 优化脚本
│   │   ├── bayesian_optimization.py
│   │   └── bayesian_optimization_quick.py
│   └── analysis/            # 分析脚本
│       ├── crypto_analysis/ # 加密货币分析
│       ├── crypto_new_analyzer/ # 新闻分析器
│       ├── price_analyzer/  # 价格分析器
│       ├── stock_analysis/  # 股票分析
│       ├── test*.py         # 测试脚本
│       └── read_pt_params.py # 参数读取
├── data/                    # 数据文件
│   ├── raw/                 # 原始数据
│   │   └── datafiles/       # 价格数据文件
│   └── processed/           # 处理后的数据
├── experiments/             # 实验相关
│   ├── cache/               # 缓存文件
│   │   ├── *.pt            # 训练好的模型
│   │   ├── classification/ # 分类任务缓存
│   │   ├── regression/     # 回归任务缓存
│   │   └── prediction_results/ # 预测结果
│   └── results/             # 实验结果
│       └── optimization_results/ # 优化结果
└── docs/                    # 文档
```

## 主要功能

### 1. 模型类型
- **TimeMixer + GCN**: 时间序列混合模型结合图卷积网络
- **LSTM + GCN**: 长短期记忆网络结合图卷积网络
- **TimeXer**: 时间序列预测模型
- **UnifiedMultiScale**: 统一多尺度模型

### 2. 任务类型
- **分类任务**: 价格涨跌分类预测
- **回归任务**: 价格数值预测

### 3. 数据源
- 加密货币价格数据
- 新闻情感分析数据
- 股票相关数据

### 4. 特性
- 支持有/无新闻数据的训练
- 支持有/无图神经网络的训练
- 贝叶斯优化超参数调优
- 多种评估指标

## 使用方法

### 1. 基本训练
```bash
python main.py --task_name regression --data_path 1H.csv
```

### 2. 使用特定模型训练
```bash
python scripts/training/train_timexer_gcn.py
```

### 3. 超参数优化
```bash
python scripts/optimization/bayesian_optimization.py
```

### 4. 模型测试
```bash
python scripts/analysis/test_timexer.py
```

## 配置说明

主要配置参数在 `config.py` 中：
- `batch_size`: 批次大小
- `seq_len`: 输入序列长度
- `pred_len`: 预测序列长度
- `threshold`: 相关系数阈值
- 各种模型特定参数

## 依赖环境

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas
- Scikit-learn
- Optuna (用于贝叶斯优化)

## 注意事项

1. 训练前确保数据文件在正确位置
2. 根据硬件配置调整批次大小
3. 不同模型可能需要不同的数据预处理
4. 缓存文件较大，注意存储空间

## 实验结果

实验结果和模型文件保存在 `experiments/` 目录下：
- `cache/`: 训练好的模型权重
- `results/`: 优化结果和性能指标
