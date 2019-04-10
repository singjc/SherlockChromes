import numpy as np
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

class XceptionConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        depth_multiplier=1):
        super(XceptionConv1dBlock, self).__init__()
        self.block = nn.ModuleList()

        for i in range(3):
            self.block.append(
                nn.Sequential(
                    DepthSeparableConv1d(
                        in_channels,
                        in_channels
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(in_channels)
                )
            )

        self.block = nn.Sequential(*self.block)

        self.residual = nn.Sequential()

    def forward(self, x):
        out = self.block(x)
        out+= self.residual(x)

        return out

class ResNextConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        cardinality=32,
        bottleneck_width=4,
        stride=1,
        expansion_factor=1):
        super(ResNextConv1dBlock, self).__init__()
        group_width = cardinality * bottleneck_width

        self.compressor = nn.Sequential(
            nn.Conv1d(
                in_channels,
                group_width,
                1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm1d(group_width)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                group_width,
                group_width,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
                groups=cardinality,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm1d(group_width)
        )

        self.expander = nn.Sequential(
            nn.Conv1d(
                group_width,
                in_channels * expansion_factor,
                1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels * expansion_factor)
        )

        self.residual = nn.Sequential()
        
        if expansion_factor > 1:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels * expansion_factor,
                    1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(in_channels * expansion_factor)
            )

    def forward(self, x):
        out = self.compressor(x)
        out = self.bottleneck(out)
        out = self.expander(out)
        out+= self.residual(x)

        return out

class SqueezeExcitationDepthSeparableResNextConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        bottleneck_width=128,
        stride=1,
        expansion_factor=1):
        super(SqueezeExcitationDepthSeparableResNextConv1dBlock, self).__init__()

        self.compressor = nn.Sequential(
            DepthSeparableConv1d(
                in_channels,
                bottleneck_width,
                1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_width)
        )

        self.bottleneck = nn.Sequential(
            DepthSeparableConv1d(
                bottleneck_width,
                bottleneck_width,
                kernel_size=kernel_size,
                stride=stride
            ),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_width)
        )

        self.expander = nn.Sequential(
            DepthSeparableConv1d(
                bottleneck_width,
                in_channels * expansion_factor,
                1,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels * expansion_factor)
        )

        self.se = SqueezeExcitation1d(in_channels * expansion_factor)

        self.residual = nn.Sequential()
        
        if expansion_factor > 1:
            self.residual = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels * expansion_factor,
                    1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(in_channels * expansion_factor)
            )

    def forward(self, x):
        out = self.compressor(x)
        out = self.bottleneck(out)
        out = self.expander(out)
        out = self.se(out)
        out+= self.residual(x)

        return out