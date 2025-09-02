import torch
import torch.nn as nn
from models.BaseModel.timemixer import TimemixerFeatureExtractor
from models.BaseModel.gcn import GCN
from models.layers.standardnorm import Normalize
import torch.nn.functional as F

class TimeMixerGCN(nn.Module):
    def __init__(self, configs, hidden_dim, output_dim):
        super(TimeMixerGCN, self).__init__()

        self.configs = configs
        self.gcn_dim = output_dim  # 新增，GCN输出特征维度
        # TimeMixer特征提取器
        self.timemixer = TimemixerFeatureExtractor(configs)
        
        # 多尺度GCN
        self.gcn_layers = nn.ModuleList([
            GCN(
                input_dim=configs.d_model,    # TimeMixer每个尺度的特征维度
                hidden_dim=hidden_dim,               # GCN隐藏层维度
                output_dim=output_dim                # GCN输出维度
            )
            for _ in range(configs.down_sampling_layers + 1)  # 对应每个尺度
        ])
        # predict_layers: 对特征做降维
        self.predict_layers = torch.nn.ModuleList([
            nn.Linear(self.gcn_dim, 64)
            for _ in range(configs.down_sampling_layers + 1)
        ])

        self.projection_layer = nn.Linear(configs.d_model, configs.c_out)

        # if self.configs.task_name == 'classification':
        #     self.act = F.gelu
        #     self.dropout = nn.Dropout(configs.dropout)
        #     self.projection = nn.Linear(
        #         configs.d_model * configs.seq_len, 2)
        
        # MLP预测头
        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),  # 添加BatchNorm  # or output_dim if用GCN
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # 添加BatchNorm  # or output_dim if用GCN
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
            if self.configs.down_sampling_method == 'max':
                down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
            elif self.configs.down_sampling_method == 'avg':
                down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
            elif self.configs.down_sampling_method == 'conv':
                padding = 1 if torch.__version__ >= '1.5.0' else 2
                down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                    kernel_size=3, padding=padding,
                                    stride=self.configs.down_sampling_window,
                                    padding_mode='circular',
                                    bias=False)
            else:
                return x_enc, x_mark_enc
            # B,T,C -> B,C,T
            x_enc = x_enc.permute(0, 2, 1)

            x_enc_ori = x_enc
            x_mark_enc_mark_ori = x_mark_enc

            x_enc_sampling_list = []
            x_mark_sampling_list = []
            x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
            x_mark_sampling_list.append(x_mark_enc)

            for i in range(self.configs.down_sampling_layers):
                x_enc_sampling = down_pool(x_enc_ori)

                x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
                x_enc_ori = x_enc_sampling

                if x_mark_enc is not None:
                    x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                    x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

            x_enc = x_enc_sampling_list
            x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

            return x_enc, x_mark_enc
        
    def forward(self, x_enc, x_mark_enc, edge_index):
        batch_size = x_enc.size(0)
        n_vars = x_enc.size(-1)
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.configs.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)
        # 1. TimeMixer特征提取 - 得到多尺度特征列表
        features_list = self.timemixer(x_list)  # 列表，每个元素shape: [B*N, S, d_model]
        # 2. GCN批量处理（已注释，直接用TimeMixer输出）
        # gcn_outputs_all_scales = []
        # for scale_idx, features in enumerate(features_list):
        #     # features: [B*N, S, d_model]
        #     S = features.size(1)
        #     d_model = features.size(2)
        #     # reshape回 [B, N, S, d_model]
        #     features_reshaped = features.view(batch_size, n_vars, S, d_model)
        #     # 合并 batch、节点、时间步为 [B*S*N, d_model]
        #     features_for_gcn = features_reshaped.permute(0, 2, 1, 3).reshape(batch_size * S * n_vars, d_model)
        #     # 构造批量edge_index
        #     edge_index_batch_list = []
        #     for b in range(batch_size):
        #         for s in range(S):
        #             offset = (b * S + s) * n_vars
        #             edge_index_batch_list.append(edge_index + offset)
        #     edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)  # [2, num_edges * B * S]
        #     # 批量GCN
        #     gcn_out = self.gcn_layers[scale_idx](features_for_gcn, edge_index_for_gcn)  # [B*S*N, gcn_dim]
        #     # reshape回 [B, N, S, gcn_dim]
        #     gcn_out = gcn_out.view(batch_size, S, n_vars, -1).permute(0, 2, 1, 3)  # [B, N, S, gcn_dim]
        #     gcn_outputs_all_scales.append(gcn_out)
        # 3. 合并所有尺度的特征（直接用TimeMixer输出features_list）
        dec_out_list = []
        for i, features in enumerate(features_list):
            # features: [B*N, S, d_model]
            B_N, S, d_model = features.shape
            B = batch_size
            N = n_vars
            # reshape为 [B, N, S, d_model]
            features_reshaped = features.view(B, N, S, d_model)
            enc_out_flat = features_reshaped.reshape(B * N, S, d_model)
            dec_out = self.predict_layers[i](enc_out_flat.mean(dim=1))  # [B*N, 64]
            dec_out = dec_out.view(B, N, -1)
            dec_out_list.append(dec_out)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # dec_out: [B, N, 64]
        # 4. MLP预测
        dec_out_flat = dec_out.reshape(-1, dec_out.size(-1))  # [B * n_vars, 64]
        logits_flat = self.mlp(dec_out_flat)  # [B * n_vars, 2]
        output = logits_flat.view(dec_out.size(0), dec_out.size(1), -1)  # [B, n_vars, 2]
        return output
    
    # def forward(self, x_enc, x_mark_enc, edge_index):
    #     batch_size = x_enc.size(0)
    #     n_vars = x_enc.size(-1)
    #     x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
    #     x_list = []
    #     for i, x in zip(range(len(x_enc)), x_enc):
    #         B, T, N = x.size()
    #         x = self.normalize_layers[i](x, 'norm')
    #         if self.configs.channel_independence:
    #             x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
    #         x_list.append(x)
    #     # 1. TimeMixer特征提取 - 得到多尺度特征列表
    #     features_list = self.timemixer(x_list)  # 列表，每个元素shape: [B*N, S, d_model]
    #     # 2. 对每个尺度分别进行GCN批量处理
    #     gcn_outputs_all_scales = []
    #     for scale_idx, features in enumerate(features_list):
    #         # features: [B*N, S, d_model]
    #         S = features.size(1)
    #         d_model = features.size(2)
    #         # reshape回 [B, N, S, d_model]
    #         features_reshaped = features.view(batch_size, n_vars, S, d_model)
    #         # 合并 batch、节点、时间步为 [B*S*N, d_model]
    #         features_for_gcn = features_reshaped.permute(0, 2, 1, 3).reshape(batch_size * S * n_vars, d_model)
    #         # 构造批量edge_index
    #         edge_index_batch_list = []
    #         for b in range(batch_size):
    #             for s in range(S):
    #                 offset = (b * S + s) * n_vars
    #                 edge_index_batch_list.append(edge_index + offset)
    #         edge_index_for_gcn = torch.cat(edge_index_batch_list, dim=1)  # [2, num_edges * B * S]
    #         # 批量GCN
    #         gcn_out = self.gcn_layers[scale_idx](features_for_gcn, edge_index_for_gcn)  # [B*S*N, gcn_dim]
    #         # reshape回 [B, N, S, gcn_dim]
    #         gcn_out = gcn_out.view(batch_size, S, n_vars, -1).permute(0, 2, 1, 3)  # [B, N, S, gcn_dim]
    #         gcn_outputs_all_scales.append(gcn_out)
    #     # 3. 合并所有尺度的特征
    #     dec_out_list = []
    #     for i, enc_out in enumerate(gcn_outputs_all_scales):
    #         # enc_out: [B, N, S, gcn_dim]
    #         B, N, S, gcn_dim = enc_out.shape
    #         enc_out_flat = enc_out.reshape(B * N, S, gcn_dim)
    #         dec_out = self.predict_layers[i](enc_out_flat.mean(dim=1))  # [B*N, 64]
    #         dec_out = dec_out.view(B, N, -1)
    #         dec_out_list.append(dec_out)
    #     dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
    #     # dec_out: [B, N, 64]
    #     # 4. MLP预测
    #     dec_out_flat = dec_out.reshape(-1, dec_out.size(-1))  # [B * n_vars, 64]
    #     logits_flat = self.mlp(dec_out_flat)  # [B * n_vars, 2]
    #     output = logits_flat.view(dec_out.size(0), dec_out.size(1), -1)  # [B, n_vars, 2]
    #     return output

