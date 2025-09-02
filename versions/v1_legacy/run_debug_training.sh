#!/bin/bash

echo "🔍 开始调试训练问题..."

# 检查conda环境
echo "=== 检查conda环境 ==="
conda env list

# 尝试不同的环境
environments=("cnn" "tf" "base")

for env in "${environments[@]}"; do
    echo ""
    echo "=== 尝试环境: $env ==="
    
    if [[ "$env" == "cnn" ]]; then
        source /mnt/nvme1n1/ly/conda_envs/cnn/bin/activate
    elif [[ "$env" == "tf" ]]; then
        conda activate tf
    else
        conda activate base
    fi
    
    # 检查PyTorch是否可用
    python -c "
try:
    import torch
    print(f'✅ PyTorch版本: {torch.__version__}')
    print(f'✅ CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    
    # 运行调试训练
    print('')
    print('🚀 开始运行调试训练...')
    exec(open('debug_training.py').read())
    
except ImportError as e:
    print(f'❌ PyTorch导入失败: {e}')
except Exception as e:
    print(f'❌ 运行失败: {e}')
    import traceback
    traceback.print_exc()
" 2>&1
    
    # 如果成功，退出循环
    if [ $? -eq 0 ]; then
        echo "✅ 在环境 $env 中成功运行"
        break
    else
        echo "❌ 在环境 $env 中失败"
    fi
done

echo ""
echo "🔧 如果所有环境都失败，请检查:"
echo "1. PyTorch是否正确安装"
echo "2. CUDA驱动是否兼容"
echo "3. 是否有足够的GPU内存"
echo "4. 数据文件是否存在"
