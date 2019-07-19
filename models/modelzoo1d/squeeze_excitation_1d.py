import torch
import torch.nn as nn

class SqueezeExcitation1d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation1d, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        out = x * y.expand_as(x)

        return out
