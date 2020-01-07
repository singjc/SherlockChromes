import torch
import torch.nn as nn

class ChannelGate1d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelGate1d, self).__init__()
        self.squeeze_avg = nn.AdaptiveAvgPool1d(1)
        self.squeeze_max = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.sigmoid(
                self.excitation(
                    self.squeeze_avg(x).view(b, c)).view(b, c, 1) + 
                self.excitation(
                    self.squeeze_max(x).view(b, c)).view(b, c, 1)
            )
        out = x * y.expand_as(x)

        return out

class SpatialGate1d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialGate1d, self).__init__()
        self.squeeze_avg = nn.AdaptiveAvgPool1d(1)
        self.squeeze_max = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(
                2,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, l = x.size()
        permuted_x = x.permute(0, 2, 1)
        y = self.excitation(
                torch.cat(
                    (
                        self.squeeze_avg(permuted_x).permute(0, 2, 1),
                        self.squeeze_max(permuted_x).permute(0, 2, 1)
                    ),
                    dim=1
                )
            )
        out = x * y

        return out

class ConvolutionalBlockAttention1d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttention1d, self).__init__()
        self.channelwise = ChannelGate1d(in_channels, reduction_ratio)
        self.spatialwise = SpatialGate1d(kernel_size)

    def forward(self, x):
        out = self.channelwise(x)
        out = self.spatialwise(out)

        return out
