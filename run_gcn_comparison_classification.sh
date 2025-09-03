#!/bin/bash
# GCN配置对比测试启动脚本 - 分类任务版本
#
# 这个脚本会自动测试6种不同的配置用于分类任务：
# 1. no_gcn - 不使用GCN（基准）
# 2. basic - 基础GCN
# 3. improved_light - 轻量级改进GCN
# 4. improved_gelu - GELU激活改进GCN
# 5. gat_attention - 图注意力网络
# 6. adaptive - 自适应GCN
#
# 任务类型: 分类 (价格变化方向预测)
# 每个配置训练20轮，预计总时间2-4小时

echo "🚀 开始自动化GCN配置对比测试 - 分类任务"
echo "📊 任务类型: 分类 (价格变化方向预测)"
echo "📋 将测试6种配置，每种配置训练20轮"
echo "⏱️ 预计总时间: 2-4小时"
echo ""

# 确保在正确的目录
cd /mnt/nvme1n1/ly/Project1

# 运行测试脚本
python scripts/experiments/auto_gcn_comparison_classification.py

echo ""
echo "✅ 分类任务GCN配置对比测试完成！"
echo "📁 结果保存在: experiments/gcn_comparison_results/"
