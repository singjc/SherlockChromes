import torch
import torch.nn as nn

class DepthwiseAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super(DepthwiseAttention1d, self).__init__()
        self.squeeze_avg = nn.AdaptiveAvgPool1d(1)
        self.squeeze_max = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            DepthSeparableConv1d(
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
        out = x + y

        return out

class ChannelwiseAttention1d(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelwiseAttention1d, self).__init__()
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
        out = x + y.expand_as(x)

        return out

class AttendedDepthSeparableConv1d(nn.Module):
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
        intermediate_nonlinearity=False,
        depthwise_attn_kernel_size=7,
        reduction_ratio=16):
        super(AttendedDepthSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)

        self.depthwise_attn = DepthwiseAttention1d(
            kernel_size=depthwise_attn_kernel_size
        )

        self.intermediate_nonlinearity = intermediate_nonlinearity

        if self.intermediate_nonlinearity:
            self.nonlinear_activation = nn.ReLU()

        self.pointwise = nn.Conv1d(
            in_channels * depth_multiplier,
            out_channels,
            1,
            bias=bias)

        self.channelwise_attn = ChannelwiseAttention1d(
            in_channels=out_channels,
            reduction_ratio=reduction_ratio)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.depthwise_attn(out)

        if self.intermediate_nonlinearity:
            out = self.nonlinear_activation(out)

        out = self.pointwise(out)
        out = self.channelwise_attn(out)

        return out

class XceptionConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        depth_multiplier=1,
        intermediate_nonlinearity=False):
        super(XceptionConv1dBlock, self).__init__()
        self.block = nn.ModuleList()

        for i in range(3):
            self.block.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels,
                        in_channels,
                        depth_multiplier=depth_multiplier,
                        intermediate_nonlinearity=intermediate_nonlinearity
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
