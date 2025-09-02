import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models.MixModel.lstm_gcn import LstmGcn
from dataloader.gnn_loader import load_gnn_data, create_gnn_dataloaders

file_path = 'Project1/datafiles/1H.csv'
input_dim = 512
hidden_dim = 128
output_dim = 32
batch_size = 32
epochs = 10
threshold = 0.6

# num_nodes = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
coin_features_list, coin_labels_list, edge_index = load_gnn_data(
    file_path=file_path,
    input_dim=input_dim,
    threshold=threshold,
    task="classification",
    norm_type='standard'
)
coin_train_loaders, coin_test_loaders = create_gnn_dataloaders(
    coin_features_list,
    coin_labels_list,
    batch_size=batch_size,
    test_size=0.2,
    task="classification"
)
    
    # 初始化模型
model = LstmGcn(1, hidden_dim, output_dim).to(device)
    
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5, verbose=True
)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    # tqdm 外层显示 Epoch
    batch_iterator = zip(*coin_train_loaders)
    num_batches = min(len(loader) for loader in coin_train_loaders)  # 最小的 batch 数量作为迭代次数
    pbar = tqdm(batch_iterator, total=num_batches, desc=f"Epoch {epoch+1}/{epochs}")

    for batches in pbar:
        coin_data_list = []
        target_list = []

        for data, target in batches:
            coin_data_list.append(data.to(device))
            target_list.append(target.to(device))

        input_data = torch.stack(coin_data_list, dim=1)
        target = torch.stack(target_list, dim=1)

        optimizer.zero_grad()
        output = model(input_data, edge_index=edge_index.to(device))
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
    print(f"✅ Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.4f}, Total Loss: {epoch_loss:.4f}，LR: {current_lr:.6f}")

model.eval()  # 切换到评估模式
total_correct = 0
total_samples = 0
total_loss = 0.0

# 初始化总计数器
total_target_zeros = 0
total_target_ones = 0
total_pred_zeros = 0
total_pred_ones = 0

# tqdm 外层显示 Epoch
batch_iterator = zip(*coin_test_loaders)
num_batches = min(len(loader) for loader in coin_test_loaders)  # 最小的 batch 数量作为迭代次数
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
        output = model(input_data, edge_index=edge_index.to(device))  # 输出形状 [batch_size, num_coins, num_classes]
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