import torch
import torch.nn as nn
from models.BaseModel.timexer import TimeXerFeatureExtractor
from models.BaseModel.gcn import GCN


class TimexerGCN(nn.Module):
    def __init__(self, configs, hidden_dim, output_dim):
        super(TimexerGCN, self).__init__()
        self.timexer = TimeXerFeatureExtractor(configs)  # 特征提取器
        # 用于节点特征的 GCN 网络
        self.gcn = GCN(configs.d_model, hidden_dim, output_dim) # GCN 模块
        # MLP 网络，用于对每个币的涨跌进行预测
        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # 添加BatchNorm  # or output_dim if用GCN
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # 添加BatchNorm  # or output_dim if用GCN
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3),
            nn.Linear(128, 2)
        )

    # def forward(self, x_enc, x_mark_enc, edge_index):
    #     # x_enc = x_enc.permute(0, 2, 1)
    #     # 1. 提取时序特征 [B, n_vars, d_model, patch_num]
    #     enc_out = self.timexer(x_enc, x_mark_enc)

    #     # 2. 平均池化 patch 维度，得到节点特征 [B, n_vars, d_model]
    #     node_features = enc_out.mean(dim=-1) 

    #     logits_list = []
    #     for i in range(node_features.size(0)):
    #         x = node_features[i]  # [n_vars, d_model]
    #         # gcn_out = self.gcn(x, edge_index)  # [n_vars, gcn_output_dim]
    #         logits = self.mlp(x)  # [n_vars, 2]
    #         logits_list.append(logits)

    #     output = torch.stack(logits_list, dim=0)  # [B, n_vars, 2]
    #     return output
    
    def forward(self, x_enc, x_mark_enc, edge_index):
        B, _, N = x_enc.size()
        enc_out = self.timexer(x_enc, x_mark_enc)

        # 2. 平均池化 patch 维度，得到节点特征 [B, n_vars, d_model]
        node_features = enc_out.mean(dim=-1) 

        # 3. 原GCN处理部分（已注释）
        # # 构造批量GCN的edge_index：对每个batch样本，将edge_index整体平移offset，拼接成大图
        # edge_index_batch_list = []
        # for i in range(B):
        #     offset = i * N  # 每个batch样本的节点编号偏移
        #     edge_index_batch_list.append(edge_index + offset)
        # edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)    # [2, num_edges * B]
        # # GCN输入：将节点特征展平成[B*N, d_model]，与批量edge_index一起送入GCN
        # gcn_input = node_features.view(B * N, -1)  # [B*N, d_model]
        # gcn_out = self.gcn(gcn_input, edge_index_for_gcn)  # [B*N, output_dim]

        # 4. 直接使用节点特征进行预测（跳过GCN）
        node_features_flat = node_features.view(B * N, -1)  # [B*N, d_model]
        mlp_predictions_flat = self.mlp(node_features_flat)  # Shape: [B*N, num_classes]

        # 5. 恢复batch结构 [B, N, num_classes]
        output = mlp_predictions_flat.view(B, N, -1)

        return output
    
    # def forward(self, x_enc, x_mark_enc, edge_index):
        # B, _, N = x_enc.size()
        # enc_out = self.timexer(x_enc, x_mark_enc)

        # # 2. 平均池化 patch 维度，得到节点特征 [B, n_vars, d_model]
        # node_features = enc_out.mean(dim=-1) 

        # edge_index_batch_list = []
        # for i in range(B):
        #     offset = i * N
        #     edge_index_batch_list.append(edge_index + offset)
        # edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)    

        # gcn_input = node_features.view(B * N, -1)
        # gcn_out = self.gcn(gcn_input, edge_index_for_gcn)

        # mlp_predictions_flat = self.mlp(gcn_out)  # Shape: [B*N, num_classes]

        # output = mlp_predictions_flat.view(B, N, -1)

        # return output



