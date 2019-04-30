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
        self.layernorm = None
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

class CustomConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        bottleneck_channels,
        attn='cbam',
        reduction_ratio=16,
        cbam_kernel_size=7):
        super(CustomConv1dBlock, self).__init__()
        self.compressor = nn.Sequential(
            nn.Conv1d(
                in_channels,
                bottleneck_channels,
                1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_channels)
        )

        self.bottleneck = nn.Sequential(
            DepthSeparableConv1d(
                bottleneck_channels,
                bottleneck_channels
            ),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_channels)
        )

        self.expander = nn.Sequential(
            nn.Conv1d(
                bottleneck_channels,
                in_channels,
                1,
                bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels)
        )

        if attn == 'cbam':
            self.attn = ConvolutionalBlockAttention1d(
                in_channels,
                reduction_ratio=reduction_ratio,
                kernel_size=cbam_kernel_size
            )
        elif attn == 'gc':
            self.attn = GlobalContextBlock1d(
                in_channels,
                reduction_ratio=reduction_ratio
            )
        else:
            self.attn = nn.Sequential()

    def forward(self, x):
        out = self.compressor(x)
        out = self.bottleneck(out)
        out = self.expander(out)
        out = self.attn(out)
        out+= x

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
                    DepthSeparableConv1d(
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

class ConvolutionalBlockAttentionXceptionConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        depth_multiplier=1,
        intermediate_nonlinearity=False,
        reduction_ratio=16,
        cbam_kernel_size=7):
        super(ConvolutionalBlockAttentionXceptionConv1dBlock, self).__init__()
        self.block = nn.ModuleList()

        for i in range(3):
            self.block.append(
                nn.Sequential(
                    DepthSeparableConv1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        depth_multiplier=depth_multiplier,
                        intermediate_nonlinearity=intermediate_nonlinearity
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(in_channels)
                )
            )

        self.block = nn.Sequential(*self.block)

        self.cbam = ConvolutionalBlockAttention1d(
            in_channels,
            reduction_ratio=reduction_ratio,
            kernel_size=cbam_kernel_size
        )

        self.residual = nn.Sequential()

    def forward(self, x):
        out = self.block(x)
        out = self.cbam(out)
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

class ConvolutionalBlockAttentionResNextConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        cardinality=32,
        bottleneck_width=4,
        stride=1,
        expansion_factor=1,
        reduction_ratio=16,
        cbam_kernel_size=7):
        super(ConvolutionalBlockAttentionResNextConv1dBlock, self).__init__()
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

        self.cbam = ConvolutionalBlockAttention1d(
            in_channels * expansion_factor,
            reduction_ratio=reduction_ratio,
            kernel_size=cbam_kernel_size
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
        out = self.cbam(out)
        out+= self.residual(x)

        return out
