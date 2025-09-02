import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models.MixModel.timemixer_gcn_no_news import TimeMixerGNN
from dataloader.gnn_loader import load_gnn_data, create_gnn_dataloaders
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--file_path', type=str, default='Project1/datafiles/1H.csv', help='数据文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--test_size', type=float, default=0.2, help='测试集比例')
    parser.add_argument('--threshold', type=float, default=0.6, help='相关系数阈值')
    
    # 模型参数
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--seq_len', type=int, default=180, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=1, help='预测长度')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='下采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='下采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='下采样方法')
    parser.add_argument('--use_norm', type=int, default=0, help='是否使用归一化')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout率')
    parser.add_argument('--channel_independence', type=bool, default=False, help='通道独立性')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn,FeedForward 网络的隐藏层维度')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')

    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    args = parser.parse_args()
    args.task_name = 'classification'  # 设置任务类型为分类
    return args

def train_epoch(args, model, train_loader, criterion, optimizer, scheduler, edge_index, device):
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        # tqdm 外层显示 Epoch
        batch_iterator = zip(*train_loader)
        num_batches = min(len(loader) for loader in train_loader)  # 最小的 batch 数量作为迭代次数
        pbar = tqdm(batch_iterator, total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batches in pbar:
            coin_data_list = []
            target_list = []

            for data, target in batches:
                coin_data_list.append(data.to(device))
                target_list.append(target.to(device))

            input_data = torch.stack(coin_data_list, dim=1)
            target = torch.stack(target_list, dim=1)

            optimizer.zero_grad()
            output = model(x_enc=input_data, x_mark_enc=None, edge_index=edge_index.to(device))
            #output = model(input_data, None)
            output_flat = output.view(-1, 2)
            target_flat = target.long().view(-1)

            loss = criterion(output_flat, target_flat)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())  # 每个 batch 更新一次 loss

        avg_loss = epoch_loss / num_batches
        # 添加scheduler步进
        scheduler.step(avg_loss)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"✅ Epoch {epoch+1}/{args.epochs} completed. Avg Loss: {avg_loss:.4f}, Total Loss: {epoch_loss:.4f}，LR: {current_lr:.6f}")

def evaluate(model, test_loader, criterion, edge_index, device):
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    # 初始化总计数器
    total_target_zeros = 0
    total_target_ones = 0
    total_pred_zeros = 0
    total_pred_ones = 0

    # tqdm 外层显示 Epoch
    batch_iterator = zip(*test_loader)
    num_batches = min(len(loader) for loader in test_loader)  # 最小的 batch 数量作为迭代次数
    pbar = tqdm(batch_iterator, total=num_batches, desc="Testing")

    with torch.no_grad():  # 测试时不计算梯度
        for batches in pbar:
            coin_data_list = []
            target_list = []

            for data, target in batches:
                coin_data_list.append(data.to(device))
                target_list.append(target.to(device))

            # 堆叠成 [batch_size, num_coins, input_dim]
            input_data = torch.stack(coin_data_list, dim=1)
            target = torch.stack(target_list, dim=1)

            # 前向传播
            output = model(x_enc=input_data, x_mark_enc=None, edge_index=edge_index.to(device))  # 输出形状 [batch_size, num_coins, num_classes]
            #output = model(input_data, None)
            output_flat = output.view(-1, 2)
            target_flat = target.long().view(-1)

            # 计算损失
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()

            # 准确率计算
            preds = torch.argmax(output_flat, dim=1)
            correct = (preds == target_flat).sum().item()

            total_correct += correct
            total_samples += target_flat.size(0)

            # 本 batch 中 0、1 的数量
            target_zeros = (target_flat == 0).sum().item()
            target_ones = (target_flat == 1).sum().item()
            pred_zeros = (preds == 0).sum().item()
            pred_ones = (preds == 1).sum().item()

            # 累加总数
            total_target_zeros += target_zeros
            total_target_ones += target_ones
            total_pred_zeros += pred_zeros
            total_pred_ones += pred_ones

            # 更新进度条的后缀，显示批次的损失、目标和预测中 0 和 1 的数量
            pbar.set_postfix(loss=loss.item())

        # 计算平均损失
        avg_loss = total_loss / num_batches

    # 输出总测试集结果
    accuracy = total_correct / total_samples
    print(f"\n✅ Test completed.")
    print(f"📊 Avg Loss: {avg_loss:.4f}, Total Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"🔢 Total Target - 0s: {total_target_zeros}, 1s: {total_target_ones}")
    print(f"🔢 Total Predicted - 0s: {total_pred_zeros}, 1s: {total_pred_ones}")

def main():
    # 获取参数
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    coin_features_list, coin_labels_list, edge_index = load_gnn_data(
        file_path=args.file_path,
        input_dim=args.seq_len,
        threshold=args.threshold,
        task=args.task_name,
        norm_type='standard'
    )
    coin_train_loaders, coin_test_loaders = create_gnn_dataloaders(
        coin_features_list,
        coin_labels_list,
        batch_size=args.batch_size,
        test_size=args.test_size,
        task=args.task_name
    )
    
    # 初始化模型
    model = TimeMixerGNN(args).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    
    # 早停设置
    best_loss = float('inf')
    patience_counter = 0
    best_model = None

    # 训练
    train_epoch(args, model, coin_train_loaders, criterion, optimizer, scheduler, edge_index, device)
    
    # 验证
    evaluate(model, coin_test_loaders, criterion, edge_index, device)
        


if __name__ == '__main__':
    main()