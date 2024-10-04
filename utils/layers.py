import torch
import torch.nn as nn

from utils.train_utils import weight_init

class MLPLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x):
        return self.mlp(x)