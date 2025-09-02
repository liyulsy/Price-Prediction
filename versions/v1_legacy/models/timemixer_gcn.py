import torch
import torch.nn as nn
from models.BaseModel.timemixer import TimemixerFeatureExtractor
from models.BaseModel.gcn import GCN
from models.layers.standardnorm import Normalize
import torch.nn.functional as F

class TimeMixerGCN(nn.Module):
    def __init__(self, configs, hidden_dim, output_dim, news_feature_dim, news_processed_dim=32, num_classes=2):
        super(TimeMixerGCN, self).__init__()
        self.configs = configs

        # TimeMixer特征提取器
        self.timemixer = TimemixerFeatureExtractor(configs)

        # 新闻特征处理MLP
        self.news_processor = nn.Sequential(
            nn.Linear(news_feature_dim, news_processed_dim * 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3),
            nn.Linear(news_processed_dim * 2, news_processed_dim)
        )

        # 多尺度GCN
        self.gcn_layers = nn.ModuleList([
            GCN(
                input_dim=configs.d_model + news_processed_dim,    # 拼接后特征维度
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            for _ in range(configs.down_sampling_layers + 1)
        ])
        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(output_dim, 64)
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )
        # self.projection_layer = nn.Linear(configs.d_model, configs.c_out)
        self.mlp = nn.Sequential(
            nn.Linear(configs.d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(configs.dropout if hasattr(configs, 'dropout') else 0.3),
            nn.Linear(128, num_classes)
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

    def forward(self, x_enc, x_mark_enc, edge_index, news_features):
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
        # 新闻特征处理
        processed_news = self.news_processor(news_features)  # [B, N, news_processed_dim]
        # 1. TimeMixer特征提取 - 得到多尺度特征列表
        features_list = self.timemixer(x_list)  # 列表，每个元素shape: [B*N, T', d_model]
        # 2. GCN批量处理（已注释，直接用TimeMixer+新闻特征拼接）
        # gcn_outputs_all_scales = []
        # for scale_idx, features in enumerate(features_list):
        #     # features: [B*N, T', d_model]
        #     T_prime = features.size(1)
        #     d_model = features.size(2)
        #     # reshape回 [B, N, T', d_model]
        #     features_reshaped = features.view(batch_size, n_vars, T_prime, d_model).permute(0, 2, 1, 3)  # [B, T', N, d_model]
        #     # 扩展新闻特征到 [B, T', N, news_processed_dim]
        #     news_feat_expand = processed_news.unsqueeze(1).expand(-1, T_prime, -1, -1)
        #     # 拼接
        #     features_cat = torch.cat([features_reshaped, news_feat_expand], dim=-1)  # [B, T', N, d_model+news_processed_dim]
        #     # 合并 batch 和节点维度，送入GCN
        #     features_cat_gcn = features_cat.permute(0, 2, 1, 3).reshape(batch_size * n_vars, T_prime, d_model + processed_news.size(-1))
        #     gcn_out = self.gcn_layers[scale_idx](features_cat_gcn, edge_index)  # [B*N, N, gcn_output_dim] or [B*N, gcn_output_dim]
        #     # reshape回 [B, N, gcn_output_dim]
        #     if gcn_out.dim() == 3:
        #         gcn_out = gcn_out[:, 0, :]  # 只取第一个节点输出（如GCN输出为[BN, N, out_dim]）
        #     gcn_out = gcn_out.view(batch_size, n_vars, -1)  # [B, N, gcn_output_dim]
        #     gcn_outputs_all_scales.append(gcn_out)
        # 3. 合并所有尺度的特征（直接用TimeMixer+新闻特征拼接）
        dec_out_list = []
        for i, features in enumerate(features_list):
            # features: [B*N, T', d_model]
            T_prime = features.size(1)
            d_model = features.size(2)
            # reshape回 [B, N, T', d_model]
            features_reshaped = features.view(batch_size, n_vars, T_prime, d_model).permute(0, 2, 1, 3)  # [B, T', N, d_model]
            # 扩展新闻特征到 [B, T', N, news_processed_dim]
            news_feat_expand = processed_news.unsqueeze(1).expand(-1, T_prime, -1, -1)
            # 拼接
            features_cat = torch.cat([features_reshaped, news_feat_expand], dim=-1)  # [B, T', N, d_model+news_processed_dim]
            # 直接用拼接特征做后续处理
            # 合并 batch 和节点维度，取均值后送入predict_layers
            features_cat_flat = features_cat.permute(0, 2, 1, 3).reshape(batch_size * n_vars, T_prime, d_model + processed_news.size(-1))
            dec_out = self.predict_layers[i](features_cat_flat.mean(dim=1))  # [B*N, 64]
            dec_out = dec_out.view(batch_size, n_vars, -1)
            dec_out_list.append(dec_out)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        # dec_out = self.projection_layer(dec_out)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        # dec_out = dec_out.permute(0, 2, 1)
        # 4. MLP预测
        dec_out_flat = dec_out.reshape(-1, dec_out.size(-1))
        logits_flat = self.mlp(dec_out_flat)
        output = logits_flat.view(dec_out.size(0), dec_out.size(1), -1)
        return output     
    # def forward(self, x_enc, x_mark_enc, edge_index, news_features):
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
    #     # 新闻特征处理
    #     processed_news = self.news_processor(news_features)  # [B, N, news_processed_dim]
    #     # 1. TimeMixer特征提取 - 得到多尺度特征列表
    #     features_list = self.timemixer(x_list)  # 列表，每个元素shape: [B*N, T', d_model]
    #     # 2. 对每个尺度分别进行GCN处理
    #     gcn_outputs_all_scales = []
    #     for scale_idx, features in enumerate(features_list):
    #         # features: [B*N, T', d_model]
    #         T_prime = features.size(1)
    #         d_model = features.size(2)
    #         # reshape回 [B, N, T', d_model]
    #         features_reshaped = features.view(batch_size, n_vars, T_prime, d_model).permute(0, 2, 1, 3)  # [B, T', N, d_model]
    #         # 扩展新闻特征到 [B, T', N, news_processed_dim]
    #         news_feat_expand = processed_news.unsqueeze(1).expand(-1, T_prime, -1, -1)
    #         # 拼接
    #         features_cat = torch.cat([features_reshaped, news_feat_expand], dim=-1)  # [B, T', N, d_model+news_processed_dim]
    #         # 合并 batch 和节点维度，送入GCN
    #         features_cat_gcn = features_cat.permute(0, 2, 1, 3).reshape(batch_size * n_vars, T_prime, d_model + processed_news.size(-1))
    #         gcn_out = self.gcn_layers[scale_idx](features_cat_gcn, edge_index)  # [B*N, N, gcn_output_dim] or [B*N, gcn_output_dim]
    #         # reshape回 [B, N, gcn_output_dim]
    #         if gcn_out.dim() == 3:
    #             gcn_out = gcn_out[:, 0, :]  # 只取第一个节点输出（如GCN输出为[BN, N, out_dim]）
    #         gcn_out = gcn_out.view(batch_size, n_vars, -1)  # [B, N, gcn_output_dim]
    #         gcn_outputs_all_scales.append(gcn_out)
    #     # 3. 合并所有尺度的特征
    #     dec_out_list = []
    #     for i, enc_out in enumerate(gcn_outputs_all_scales):
    #         dec_out = self.predict_layers[i](enc_out)
    #         dec_out_list.append(dec_out)
    #     dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
    #     # dec_out = self.projection_layer(dec_out)
    #     dec_out = self.normalize_layers[0](dec_out, 'denorm')
    #     # dec_out = dec_out.permute(0, 2, 1)
    #     # 4. MLP预测
    #     dec_out_flat = dec_out.reshape(-1, dec_out.size(-1))
    #     logits_flat = self.mlp(dec_out_flat)
    #     output = logits_flat.view(dec_out.size(0), dec_out.size(1), -1)
    #     return output 
