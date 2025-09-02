# MAE修改总结

## 修改概述

根据用户要求，对测试集的预测值和真实值保存方式以及MAE计算方式进行了以下修改：

## 1. 新建文件夹保存测试集预测结果

### 修改内容：
- 创建新的保存目录：`experiments/cache/test_predictions`
- 添加了 `save_test_predictions()` 函数来保存测试集的预测值和真实值

### 保存的文件：
1. **详细预测结果文件**: `test_predictions_{timestamp}.csv`
   - 包含每个样本每个币种的预测值和真实值
   - 列名：`timestamp, coin, true_value, predicted_value, absolute_error, percentage_error`

2. **统计信息文件**: `test_statistics_{timestamp}.csv`
   - 包含每个币种的统计信息
   - 列名：`coin, mean_true, mean_pred, std_true, std_pred, min_true, min_pred, max_true, max_pred, mae, mape`

## 2. MAE计算方式修改

### 原来的MAE计算方式：
```python
mae = mean_absolute_error(original_targets, original_preds)  # 平均绝对误差
```

### 新的MAE计算方式：
```python
total_true_sum = np.sum(original_targets)
total_pred_sum = np.sum(original_preds)
new_mae = total_true_sum / total_pred_sum  # 所有真实值之和除以预测值之和
```

### 指标结构修改：
- **保留原来的MAE**: `'mae'` 键仍然使用原来的计算方式
- **添加新的MAE**: `'new_mae'` 键使用新的计算方式

## 3. 修改的文件

### 主要修改文件：
1. `scripts/training/train_timexer.py`
2. `scripts/training/unified_train_script.py`
3. `scripts/training/train_multiscale.py`
4. `scripts/training/unified_lstm_train_script.py`

### 修改内容：
1. **添加导入**：
   ```python
   import csv
   from datetime import datetime
   ```

2. **添加保存函数**：
   ```python
   def save_test_predictions(all_preds, all_targets, coin_names, timestamp=None):
       # 保存预测结果到CSV文件
   ```

3. **修改evaluate_model函数**：
   - 返回值从 `metrics` 改为 `metrics, all_preds, all_targets`
   - 添加新的MAE计算方式

4. **修改测试部分**：
   - 调用保存函数保存测试集预测结果
   - 处理反归一化以获得原始尺度的预测值和真实值

## 4. 新的指标输出

测试结果现在包含以下指标：
- `mae`: 原来的MAE计算方式（平均绝对误差）
- `new_mae`: 新的MAE计算方式（真实值之和/预测值之和）
- `mse`: 均方误差
- `rmse`: 均方根误差
- `r2`: R²决定系数
- `mape`: 平均绝对百分比误差
- 其他归一化指标...

## 5. 使用示例

运行训练脚本后，会自动：
1. 在测试阶段保存预测结果到 `experiments/cache/test_predictions/` 文件夹
2. 输出包含新旧两种MAE计算结果的指标

### 输出示例：
```
✅ Test Results:
  Overall:
    - MAE: 1.6019  # 原来的MAE
    - NEW_MAE: 0.9965  # 新的MAE
    - MSE: 4.2345
    - RMSE: 2.0578
    - R2: 0.8234
    - MAPE: 12.34
```

## 6. 注意事项

1. 新的MAE计算方式 `new_mae = sum(true_values) / sum(predicted_values)` 在理想情况下应该接近1.0
2. 原来的MAE计算方式仍然保留，可以用于对比
3. 保存的预测结果文件使用时间戳命名，避免覆盖
4. 只有在回归任务时才会保存预测结果
5. 保存的预测值和真实值都是反归一化后的原始尺度数据

## 7. 测试验证

可以运行 `test_mae_modification.py` 来验证修改是否正确工作。
