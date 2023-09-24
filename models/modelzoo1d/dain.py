import numpy as np
import torch
import torch.nn as nn


class DAIN_Layer(nn.Module):
    # Adapted from https://github.com/passalis/dain/blob/master/dain.py
    def __init__(self, mode='full', input_dim=6):
        super(DAIN_Layer, self).__init__()
        self.mode = mode

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(
            data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim, n_feature_vectors)

        # Nothing to normalize
        if not self.mode:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, dim=2)
            avg = avg.unsqueeze(-1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, dim=2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.unsqueeze(-1)
            x = x - adaptive_avg

        # Perform the first + second step (adaptive averaging + scaling)
        elif self.mode == 'adaptive_scale':

            # Step 1:
            avg = torch.mean(x, dim=2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.unsqueeze(-1)
            x = x - adaptive_avg

            # Step 2:
            std = torch.mean(x ** 2, dim=2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.unsqueeze(-1)
            x = x / (adaptive_std)

        elif self.mode == 'full':

            # Step 1:
            avg = torch.mean(x, dim=2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.unsqueeze(-1)
            x = x - adaptive_avg

            # # Step 2:
            std = torch.mean(x ** 2, dim=2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1

            adaptive_std = adaptive_std.unsqueeze(-1)
            x = x / adaptive_std

            # Step 3:
            avg = torch.mean(x, dim=2)
            gate = torch.sigmoid(self.gating_layer(avg))
            gate = gate.unsqueeze(-1)
            x = x * gate

        else:
            assert False

        return x
