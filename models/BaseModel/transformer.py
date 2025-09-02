import torch
import torch.nn as nn


class TransfromerModel(nn.Module):
    def __init__(self, L1, num_class, d_model=68, num_layers=2, n_head=8, dropout=0.2):
        super().__init__()
        #self.d_embed = d_embed
        self.d_model = d_model
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                dropout=dropout)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                dropout=dropout)
        self.encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=num_layers)
        self.encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=num_layers)
        self.encoder3 = nn.Sequential(
                nn.Linear(L1, 512),
            )
        self.decoder = nn.Sequential(
                nn.Linear(512, num_class),
            )
    def forward(self, x):
        x = x.reshape(x.size(0), 2, -1, self.d_model)
        batch_size = x.shape[0]
        x1 = x[:, 0, ...]
        x2 = x[:, 1, ...]
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x1 = x1.reshape(batch_size, -1)
        x2 = x2.reshape(batch_size, -1)
        x = torch.cat([x1, x2], axis=1)
        x = self.encoder3(x)
        y = self.decoder(x)
        return y