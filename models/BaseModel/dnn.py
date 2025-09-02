import torch.nn as nn

class DnnModel(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.dnn = nn.Sequential(
                nn.Linear(input_dim, 1024),
 
                nn.Dropout(0.3),
                nn.ReLU(True),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.ReLU(True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.ReLU(True),
                nn.Linear(128, num_class)
                )

    def forward(self, x):
        out = self.dnn(x)
        return out