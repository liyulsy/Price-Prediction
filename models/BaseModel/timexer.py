import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.selfattention_family import FullAttention, AttentionLayer
from models.layers.embed import DataEmbedding_inverted, PositionalEmbedding

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
    
class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class TimeXerFeatureExtractor(nn.Module):
    def __init__(self, configs):
        super(TimeXerFeatureExtractor, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)
        self.num_time_features = getattr(configs, 'num_time_features', 6) # Default to 6 if not in configs

        self.en_embedding = EnEmbedding(configs.enc_in, configs.d_model, self.patch_len, configs.dropout)
        # 修复：DataEmbedding_inverted的输入维度应该是时间步长，不是特征数
        # 它会在内部处理时间特征的拼接
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                               configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
            ) for l in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))
        # self.head_nf = configs.d_model * (self.patch_num + 1)
        # self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
        #                         head_dropout=configs.dropout)
    
    def forward(self, x_enc, x_mark_enc):
        # x_enc shape: [batch_size, seq_len, n_vars_input]
        # x_mark_enc shape: [batch_size, seq_len, n_time_features] or None

        x_enc_for_temporal = x_enc # Keep original for ex_embedding: [B, S, N]
        x_enc_for_patching = x_enc.permute(0, 2, 1) # Permute for patching: [B, N, S]

        if self.use_norm:
            # Normalization from Non-stationary Transformer, applied per variable over its sequence
            # x_enc_for_patching shape: [B, N, S]
            means = x_enc_for_patching.mean(dim=2, keepdim=True).detach() # Mean over S (sequence dimension)
            x_enc_for_patching = x_enc_for_patching - means
            stdev = torch.sqrt(torch.var(x_enc_for_patching, dim=2, keepdim=True, unbiased=False) + 1e-5) # Var over S
            x_enc_for_patching = x_enc_for_patching / stdev
            # means and stdev are now [B, N, 1]

        # _, _, N_shape_check = x_enc_for_patching.shape # N_shape_check should be S (original seq_len)

        # 特征提取
        # en_embedding expects [B, N, S]
        en_embed, n_vars = self.en_embedding(x_enc_for_patching)
        
        # Prepare x_mark_enc for ex_embedding
        x_mark_enc_to_pass = x_mark_enc
        if x_mark_enc_to_pass is None and self.num_time_features > 0:
            batch_size = x_enc_for_temporal.shape[0]
            seq_len = x_enc_for_temporal.shape[1]
            x_mark_enc_to_pass = torch.zeros(batch_size, seq_len, self.num_time_features, 
                                               device=x_enc_for_temporal.device, dtype=x_enc_for_temporal.dtype)
        
        # ex_embedding expects x_enc as [B, S, N] and x_mark_enc as [B, S, n_time_features]
        ex_embed = self.ex_embedding(x_enc_for_temporal, x_mark_enc_to_pass)

        # 编码器输出
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.use_norm:
            # De-Normalization for enc_out [B, N, d_model, patch_num]
            # stdev and means are [B, N, 1]
            # Remove incorrect permutes for stdev and means if they were already [B,N,1]
            # stdev = stdev.permute(0, 2, 1) # This was incorrect if stdev is [B,N,1]
            # means = means.permute(0, 2, 1) # This was incorrect
            
            # Expand stdev/means [B,N,1] to [B,N,1,1] for broadcasting with enc_out [B,N,d_model,patch_num]
            stdev_for_denorm = stdev.unsqueeze(-1) # Shape [B,N,1,1]
            means_for_denorm = means.unsqueeze(-1) # Shape [B,N,1,1]
            
            # The original code was: stdev.unsqueeze(-1).expand(-1, -1, enc_out.size(2), enc_out.size(3))
            # This is fine if stdev is [B,N,1], .unsqueeze(-1) makes it [B,N,1,1] and then expand makes it [B,N,D,P]
            # Let's use the original expand as it's more explicit if dimensions of enc_out might vary in d_model or patch_num
            expanded_stdev = stdev.unsqueeze(-1).expand(-1, -1, enc_out.size(2), enc_out.size(3))
            expanded_means = means.unsqueeze(-1).expand(-1, -1, enc_out.size(2), enc_out.size(3))
            enc_out = enc_out * expanded_stdev + expanded_means

        return enc_out