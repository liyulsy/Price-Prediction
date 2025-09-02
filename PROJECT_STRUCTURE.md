# 项目结构说明

## 📁 项目整体结构

```
Project1/
├── 📂 scripts/                    # 主要脚本目录
│   ├── 📂 training/               # 训练脚本（统一版本）
│   │   ├── train_timemixer.py     # TimeMixer模型训练
│   │   ├── train_timexer.py       # TimeXer模型训练
│   │   ├── train_cnn.py           # CNN模型训练
│   │   ├── train_lstm.py          # LSTM模型训练
│   │   ├── train_multiscale.py    # 多尺度TimeMixer训练
│   │   ├── run_all_models.py      # 批量运行所有模型
│   │   └── README.md              # 训练脚本使用说明
│   ├── 📂 analysis/               # 数据分析脚本（已清理）
│   │   ├── 📂 crypto_analysis/    # 加密货币数据分析
│   │   ├── 📂 crypto_new_analyzer/ # 新闻分析器（核心功能）
│   │   └── test_advanced_graph_construction.py # 图构建分析
│   └── README.md                  # Scripts目录说明
├── 📂 models/                     # 模型定义
│   ├── 📂 BaseModel/              # 基础模型
│   │   ├── gcn.py                 # GCN基础模型
│   │   └── advanced_gcn.py        # 高级GCN模型
│   ├── 📂 MixModel/               # 混合模型（当前版本）
│   │   ├── unified_multiscale_timemixer_gcn.py  # 统一多尺度TimeMixer+GCN
│   │   ├── unified_timexer_gcn.py               # 统一TimeXer+GCN
│   │   ├── unified_cnn_gcn.py                   # 统一CNN+GCN
│   │   └── unified_lstm_gcn.py                  # 统一LSTM+GCN
│   └── 📂 layers/                 # 模型层定义
├── 📂 dataloader/                 # 数据加载器
│   ├── gnn_loader.py              # GNN数据加载（含高级图构建）
│   └── ...
├── 📂 utils/                      # 工具函数
├── 📂 experiments/                # 实验结果
│   └── 📂 cache/                  # 缓存目录
│       ├── 📂 test_predictions/   # 测试预测结果
│       ├── 📂 graph_analysis/     # 图分析结果
│       └── 📂 unified_comparison/ # 模型比较结果
├── 📂 versions/                   # 版本管理
│   └── 📂 v1_legacy/              # 旧版本文件
│       ├── 📂 training_scripts/   # 旧训练脚本
│       ├── 📂 models/             # 旧模型文件
│       ├── 📂 experimental/       # 实验性文件
│       └── MAE_MODIFICATION_SUMMARY.md
└── 📄 PROJECT_STRUCTURE.md        # 本文件
```

## 🎯 当前活跃的核心文件

### 训练脚本（推荐使用）
- `scripts/training/train_timemixer.py` - **主要推荐**，多尺度TimeMixer
- `scripts/training/train_timexer.py` - TimeXer模型
- `scripts/training/train_cnn.py` - CNN模型
- `scripts/training/train_lstm.py` - LSTM模型
- `scripts/training/run_all_models.py` - **批量比较所有模型**

### 核心模型文件
- `models/MixModel/unified_multiscale_timemixer_gcn.py` - **主要模型**
- `models/MixModel/unified_timexer_gcn.py` - TimeXer+GCN
- `models/MixModel/unified_cnn_gcn.py` - CNN+GCN
- `models/MixModel/unified_lstm_gcn.py` - LSTM+GCN

### 数据处理
- `dataloader/gnn_loader.py` - **图构建和数据加载**
- `scripts/analysis/crypto_new_analyzer/unified_dataset.py` - 统一数据集

### 图构建分析
- `scripts/analysis/test_advanced_graph_construction.py` - 图方法分析
- `models/BaseModel/advanced_gcn.py` - 高级GCN实现

## 🗂️ 版本管理说明

### v1_legacy/ 目录内容
存放项目早期版本的文件，包括：
- **旧训练脚本**: 各种实验性和早期版本的训练脚本
- **旧模型文件**: 早期的模型实现
- **实验性文件**: 调试、测试和实验性代码
- **旧文档**: 早期的文档和配置文件

### 当前版本特点
1. **统一的图构建方法**: 所有模型使用相同的图构建策略
2. **标准化的结果保存**: 使用模型名称而非时间戳命名
3. **支持边权重的GCN**: 改进的图卷积实现
4. **批量比较功能**: 自动运行和比较多个模型

## 🚀 快速开始

### 运行单个模型
```bash
# 运行推荐的TimeMixer模型
python scripts/training/train_timemixer.py

# 运行TimeXer模型
python scripts/training/train_timexer.py
```

### 批量比较所有模型
```bash
# 自动运行所有模型并生成比较报告
python scripts/training/run_all_models.py
```

### 分析图构建方法
```bash
# 分析不同图构建方法的效果
python scripts/analysis/test_advanced_graph_construction.py
```

## 📊 结果文件说明

### 模型文件命名规范
```
{ModelName}_{TaskType}_{GCN}_{News}_{model_file}.pt
例如: TimeMixer_regression_with_gcn_with_news_best_timemixer_model.pt
```

### 预测结果文件
```
test_predictions_{ModelName}_{TaskType}_{GCN}_{News}.csv
test_statistics_{ModelName}_{TaskType}_{GCN}_{News}.csv
```

### 比较结果文件
```
experiments/cache/unified_comparison/
├── detailed_comparison_YYYYMMDD_HHMMSS.csv
└── model_ranking_YYYYMMDD_HHMMSS.csv
```

## 🔧 配置说明

### 主要配置项
所有训练脚本都包含统一的配置结构：

```python
# 主要开关
TASK_TYPE = 'regression'
USE_GCN = True
USE_NEWS_FEATURES = True

# 图构建配置（基于实验结果优化）
GRAPH_METHOD = 'original'  # 推荐使用
GRAPH_PARAMS = {'threshold': 0.6}

# 训练参数
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0005
```

## 📈 实验结果总结

基于图构建方法比较实验的结论：
- **Original方法表现最佳**: MAE=0.312, R²=0.825
- **简单有效**: 适度的图密度避免噪声
- **计算高效**: 较少的边数，训练更快

## 🛠️ 维护说明

### 添加新模型
1. 在`models/MixModel/`中创建新的统一模型文件
2. 在`scripts/training/`中创建对应的训练脚本
3. 更新`run_all_models.py`中的模型列表

### 版本管理
- 重大更改前，将当前版本移动到`versions/v{x}_legacy/`
- 保持向后兼容性文档
- 更新`PROJECT_STRUCTURE.md`

---

**最后更新**: 2025-07-15
**当前版本**: v2 (统一图构建版本)
**推荐入口**: `scripts/training/run_all_models.py`
