import torch
import torch.nn as nn

class GlobalContextModeller1d(nn.Module):
    def __init__(self, in_channels=32):
        super(GlobalContextModeller1d, self).__init__()
        self.context_modeller = nn.Sequential(
            nn.Conv1d(
                in_channels,
                1,
                1
            ),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        out = self.context_modeller(x)
        out = out.permute(0, 2, 1)

        return out

class FeatureTransformer1d(nn.Module):
    def __init__(self, in_channels=32, reduction_ratio=16):
        super(FeatureTransformer1d, self).__init__()
        self.feature_transformer = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels // reduction_ratio,
                1
            ),
            nn.LayerNorm((in_channels // reduction_ratio, 1)),
            nn.ReLU(),
            nn.Conv1d(
                in_channels // reduction_ratio,
                in_channels,
                1
            )
        )
    
    def forward(self, x):
        out = self.feature_transformer(x)

        return out

class GlobalContextBlock1d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(GlobalContextBlock1d, self).__init__()
        self.context_modeller = GlobalContextModeller1d(
            in_channels=in_channels)
        self.feature_transformer = FeatureTransformer1d(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio)

    def forward(self, x):
        global_context = torch.matmul(x, self.context_modeller(x))

        transformed_features = self.feature_transformer(global_context)

        out = x + transformed_features.expand_as(x)

        return out
