# Scripts 目录说明

## 📁 目录结构

```
scripts/
├── 📂 training/                    # 训练脚本（核心功能）
│   ├── train_timemixer.py         # TimeMixer模型训练
│   ├── train_timexer.py           # TimeXer模型训练
│   ├── train_cnn.py               # CNN模型训练
│   ├── train_lstm.py              # LSTM模型训练
│   ├── train_multiscale.py        # 多尺度TimeMixer训练
│   ├── run_all_models.py          # 批量运行所有模型
│   └── README.md                  # 训练脚本使用说明
└── 📂 analysis/                   # 数据分析脚本
    ├── 📂 crypto_analysis/        # 加密货币数据分析
    │   ├── 📂 data/               # 价格数据存储
    │   │   ├── processed_data/    # 处理后的数据
    │   │   └── raw_data/          # 原始数据
    │   ├── merge_data.py          # 数据合并脚本
    │   └── README.md              # 数据分析说明
    ├── 📂 crypto_new_analyzer/    # 新闻分析器（核心功能）
    │   ├── 📂 crypto_news/        # 新闻数据
    │   ├── 📂 features/           # 提取的特征
    │   ├── 📂 models/             # 预训练模型
    │   ├── unified_dataset.py     # 统一数据集类（核心）
    │   ├── feature_extractor.py   # 特征提取器
    │   └── dataset.py             # 数据集类
    └── test_advanced_graph_construction.py  # 图构建方法测试
```

## 🎯 核心功能

### 训练脚本 (scripts/training/)
- **主要用途**: 模型训练和评估
- **推荐入口**: `run_all_models.py` - 批量运行所有模型
- **单模型训练**: 各个 `train_*.py` 脚本

### 数据分析 (scripts/analysis/)

#### crypto_analysis/
- **用途**: 加密货币价格数据的预处理和分析
- **数据存储**: 包含1小时和1天粒度的价格数据
- **处理脚本**: 数据合并、差异计算、格式转换等

#### crypto_new_analyzer/
- **用途**: 新闻数据的获取、处理和特征提取
- **核心文件**: `unified_dataset.py` - 统一的数据集接口
- **新闻数据**: 8个主要加密货币的新闻数据
- **特征提取**: 基于BERT的新闻情感分析

#### 图构建分析
- **test_advanced_graph_construction.py**: 测试和比较不同的图构建方法

## 🚀 快速开始

### 训练模型
```bash
# 批量运行所有模型
python scripts/training/run_all_models.py

# 运行单个模型
python scripts/training/train_timemixer.py
```

### 数据分析
```bash
# 测试图构建方法
python scripts/analysis/test_advanced_graph_construction.py

# 数据预处理（如需要）
python scripts/analysis/crypto_analysis/scripts/merge_data.py
```

## 📊 数据路径

### 价格数据
- **1小时数据**: `scripts/analysis/crypto_analysis/data/processed_data/1H/all_1H.csv`
- **1天数据**: `scripts/analysis/crypto_analysis/data/processed_data/1D/all_1D.csv`

### 新闻数据
- **原始新闻**: `scripts/analysis/crypto_new_analyzer/crypto_news/`
- **提取特征**: `scripts/analysis/crypto_new_analyzer/features/`

## 🔧 维护说明

### 已移动到legacy的文件
以下文件已移动到 `versions/v1_legacy/` 目录：
- `scripts_analysis/`: 旧的分析脚本
- `scripts_optimization/`: 贝叶斯优化脚本
- 其他不常用的工具脚本

### 清理的内容
- 删除了所有 `__pycache__` 目录
- 移除了重复和过时的脚本
- 保留了核心功能和数据

---

**最后更新**: 2025-07-15  
**维护状态**: 已清理和重组  
**推荐使用**: `scripts/training/run_all_models.py`
