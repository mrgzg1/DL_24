import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List
from torch.nn import functional as F


"""
Create a simple LSTM network for predicting future feature representations

Input:
- state_0: [B, enc_dim]
- actions: [B, T-1, 2]
- Input to the MLP is the concatenation of the previous feature representation and action
- The MLP predicts the future feature representation at each timestep


Output:
- pred_encs: [B, T, enc_dim]
"""        

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Predictor(nn.Module):
    def __init__(self, enc_dim, action_dim, arch, n_steps=17, norm_features=False):
        super(Predictor, self).__init__()
        self.enc_dim = enc_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.norm_features = norm_features

        arch_list = list(map(int, arch.split("-")))

        self.mlp = build_mlp([enc_dim + action_dim] + arch_list + [enc_dim])

    def forward(self, state, action):
        """
        Args:
            state_0: [B, enc_dim]
            action: [B, 2]

        Output:
            pred_encs: [B, enc_dim]
        """       

        x = torch.cat([state, action], dim=1)

        x = self.mlp(x)

        # Normalize the output
        if self.norm_features:
            x = F.normalize(x, dim=-1)

        return x