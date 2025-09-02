# 版本信息 - v1_legacy

## 📋 版本概述

这个目录包含了项目的早期版本文件，主要是在图构建方法优化之前的实现。

## 📁 目录结构

```
v1_legacy/
├── training_scripts/          # 旧版训练脚本
│   ├── train_test_*.py       # 早期测试脚本
│   ├── train_timexer.py      # 旧版TimeXer训练
│   ├── unified_*.py          # 早期统一脚本
│   └── price_prediction_unified_style.py
├── models/                   # 旧版模型文件
│   ├── cnn_gnn.py           # 早期CNN+GNN实现
│   ├── timemixer_gcn.py     # 早期TimeMixer+GCN
│   ├── timexer_gcn*.py      # 各种TimeXer变体
│   └── ...
├── experimental/            # 实验性文件
│   ├── compare_graph_methods.py
│   ├── quick_graph_comparison.py
│   ├── debug_training.py
│   ├── test_*.py
│   └── ...
├── MAE_MODIFICATION_SUMMARY.md  # 旧版MAE修改总结
├── README_old.md            # 旧版README
└── VERSION_INFO.md          # 本文件
```

## 🔄 主要变更

### v1 -> v2 的主要改进

1. **图构建方法统一**
   - v1: 各脚本使用不同的图构建方法
   - v2: 统一使用经过实验验证的`original`方法

2. **结果保存规范化**
   - v1: 使用时间戳命名结果文件
   - v2: 使用模型名称命名，便于识别和比较

3. **边权重支持**
   - v1: 基础的GCN实现
   - v2: 支持边权重的高级GCN

4. **训练脚本整合**
   - v1: 分散的训练脚本，配置不统一
   - v2: 统一的训练框架，标准化配置

5. **批量比较功能**
   - v1: 手动运行和比较模型
   - v2: 自动化的模型比较和排名

## 📊 实验发现

### 图构建方法比较结果
通过v1阶段的大量实验，我们发现：

| 方法 | 复杂度 | 性能 | 问题 |
|------|--------|------|------|
| Original | 简单 | 最佳 | 无 |
| Multi-layer | 复杂 | 中等 | 过度连接 |
| Dynamic | 中等 | 中等 | 计算复杂 |
| Domain Knowledge | 复杂 | 较差 | 引入噪声 |
| Attention-based | 复杂 | 较差 | 权重异常 |

**关键洞察**: 简单的相关性图方法表现最佳，复杂方法容易过拟合。

## 🗂️ 文件说明

### 训练脚本类别

1. **早期测试脚本** (`train_test_*.py`)
   - 用于初期模型验证
   - 配置简单，功能基础
   - 已被统一脚本替代

2. **模型特定脚本** (`train_timexer.py`, etc.)
   - 针对特定模型的训练脚本
   - 配置分散，不易维护
   - 已整合到统一框架

3. **统一尝试** (`unified_*.py`)
   - 早期的统一化尝试
   - 部分功能不完整
   - 为v2版本提供了基础

### 模型文件类别

1. **基础实现** (`cnn_gnn.py`, `timemixer_gcn.py`)
   - 早期的模型实现
   - 功能相对简单
   - 缺乏统一接口

2. **变体实验** (`timexer_gcn_*.py`)
   - 各种模型变体的尝试
   - 用于探索不同架构
   - 部分想法被整合到v2

3. **无新闻版本** (`*_no_news.py`)
   - 不使用新闻特征的版本
   - 用于对比实验
   - 功能已整合到统一模型

### 实验性文件

1. **图方法比较** (`compare_graph_methods.py`, `quick_graph_comparison.py`)
   - 用于比较不同图构建方法
   - 产生了重要的实验结果
   - 指导了v2的设计决策

2. **调试工具** (`debug_training.py`, `diagnose_training_issues.py`)
   - 用于排查训练问题
   - 帮助优化训练流程
   - 经验整合到v2

3. **测试脚本** (`test_*.py`)
   - 各种功能测试
   - 验证模型正确性
   - 为v2提供了测试基础

## 🔧 迁移指南

### 从v1迁移到v2

1. **训练脚本迁移**
   ```bash
   # v1 方式
   python scripts/training/train_timexer.py
   
   # v2 方式
   python scripts/training/train_timexer.py  # 新的统一版本
   ```

2. **配置更新**
   ```python
   # v1 配置（分散）
   USE_GCN = True
   threshold = 0.6
   
   # v2 配置（统一）
   USE_GCN = True
   GRAPH_METHOD = 'original'
   GRAPH_PARAMS = {'threshold': 0.6}
   ```

3. **结果文件**
   ```bash
   # v1 命名
   model_20250715_123456.pt
   predictions_20250715_123456.csv
   
   # v2 命名
   TimeMixer_regression_with_gcn_with_news_best_timemixer_model.pt
   test_predictions_TimeMixer_regression_with_gcn_with_news.csv
   ```

## 📈 性能对比

### v1 vs v2 主要改进

| 指标 | v1 | v2 | 改进 |
|------|----|----|------|
| 最佳MAE | 0.345 | 0.312 | ↓ 9.6% |
| 训练效率 | 基准 | +15% | 图构建优化 |
| 代码维护性 | 低 | 高 | 统一框架 |
| 结果可比性 | 差 | 优 | 标准化命名 |

## 🚫 废弃原因

### 为什么移动到legacy

1. **配置分散**: 各脚本配置不统一，难以维护
2. **命名混乱**: 时间戳命名难以识别和比较
3. **功能重复**: 多个脚本实现相似功能
4. **性能次优**: 图构建方法未经优化
5. **缺乏自动化**: 需要手动运行和比较

### 保留价值

1. **实验记录**: 保留了重要的实验过程
2. **设计演进**: 展示了项目的发展历程
3. **备份参考**: 可作为功能参考和回退选项
4. **学习资源**: 展示了不同的实现方法

## 🔮 未来计划

1. **定期清理**: 移除确认无用的文件
2. **文档完善**: 补充重要实验的详细记录
3. **精华提取**: 将有价值的想法整合到主版本
4. **版本标记**: 为重要节点创建版本标记

---

**创建时间**: 2025-07-15  
**对应主版本**: v2.0  
**状态**: 已废弃，仅供参考
