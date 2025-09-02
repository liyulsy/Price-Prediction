#!/usr/bin/env python3
"""
快速测试脚本：验证模型是否还存在单一预测问题
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.MixModel.unified_multiscale_timemixer_gcn import UnifiedMultiScaleTimeMixer
from dataloader.gnn_loader import create_gnn_dataloader

def test_prediction_diversity():
    """测试模型预测的多样性"""
    
    # 创建测试配置
    class TestConfig:
        def __init__(self):
            self.seq_len = 60
            self.pred_len = 1
            self.d_model = 256
            self.d_ff = 512
            self.num_kernels = 6
            self.top_k = 5
            self.down_sampling_layers = 2
            self.down_sampling_window = 2
            self.down_sampling_method = 'avg'
            self.channel_independence = False
            self.decomp_method = 'moving_avg'
            self.moving_avg = 25
            self.dropout = 0.1
    
    config = TestConfig()
    
    # 创建模型
    model = UnifiedMultiScaleTimeMixer(
        configs=config,
        num_features=8,  # 8个加密货币
        num_classes=3,   # 3分类
        task_type='classification',
        use_gcn=True,
        has_news=False,
        gcn_config={'type': 'GCN', 'hidden_dim': 128, 'num_layers': 2}
    )
    
    model.eval()
    
    # 创建测试数据
    batch_size = 32
    seq_len = 60
    num_features = 8
    
    x_enc = torch.randn(batch_size, seq_len, num_features)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)  # 时间特征
    
    # 创建简单的图结构
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    with torch.no_grad():
        # 多次前向传播测试
        predictions = []
        for _ in range(5):
            output = model(x_enc, x_mark_enc, edge_index=edge_index)
            predictions.append(output.cpu().numpy())
    
    # 分析预测多样性
    predictions = np.array(predictions)  # [5, batch_size, num_features, num_classes]
    
    print("🔍 预测多样性分析:")
    print(f"预测形状: {predictions.shape}")
    
    # 检查每个样本的预测分布
    for i in range(min(3, batch_size)):  # 检查前3个样本
        sample_preds = predictions[:, i, :, :]  # [5, num_features, num_classes]
        
        print(f"\n样本 {i+1}:")
        for j in range(num_features):
            feature_preds = sample_preds[:, j, :]  # [5, num_classes]
            
            # 计算预测的标准差
            pred_std = np.std(feature_preds, axis=0)
            pred_mean = np.mean(feature_preds, axis=0)
            
            # 应用softmax获得概率
            probs = F.softmax(torch.tensor(feature_preds), dim=-1).numpy()
            prob_std = np.std(probs, axis=0)
            
            print(f"  特征 {j}: 原始输出std={pred_std}, 概率std={prob_std}")
            
            # 检查是否所有预测都相同
            if np.all(pred_std < 1e-6):
                print(f"    ⚠️  特征 {j} 预测完全相同!")
            else:
                print(f"    ✅ 特征 {j} 预测有变化")
    
    # 整体统计
    all_preds_flat = predictions.reshape(-1, predictions.shape[-1])
    overall_std = np.std(all_preds_flat, axis=0)
    
    print(f"\n📊 整体统计:")
    print(f"所有预测的标准差: {overall_std}")
    print(f"预测是否有多样性: {'是' if np.any(overall_std > 1e-3) else '否'}")
    
    return predictions

if __name__ == "__main__":
    print("🧪 开始测试模型预测多样性...")
    try:
        predictions = test_prediction_diversity()
        print("\n✅ 测试完成!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
