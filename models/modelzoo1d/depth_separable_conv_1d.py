import torch
import torch.nn as nn

class DepthSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
        depth_multiplier=1,
        intermediate_nonlinearity=False):
        super(DepthSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.intermediate_nonlinearity = intermediate_nonlinearity

        if self.intermediate_nonlinearity:
            self.nonlinear_activation = nn.ReLU()

        self.pointwise = nn.Conv1d(
            in_channels * depth_multiplier,
            out_channels,
            1,
            bias=bias)

    def forward(self, x):
        out = self.depthwise(x)

        if self.intermediate_nonlinearity:
            out = self.nonlinear_activation(out)

        out = self.pointwise(out)

        return out
