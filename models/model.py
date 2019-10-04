import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '../models')

from custom_layers_and_blocks import AttendedDepthSeparableConv1d

class AtrousChannelwiseEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels=14,
        out_channels=[32, 16, 8, 4, 2],
        kernel_sizes=[9, 9, 9, 9, 9],
        dilations=[1, 1, 2, 2, 3]):
        super(AtrousChannelwiseEncoderDecoder, self).__init__()
        self.encoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=((kernel_sizes[0] - 1) // 2 * dilations[0]),
                    dilation=dilations[0],
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                    reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        for i in range(1, len(out_channels)):
            self.encoder.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=out_channels[i - 1],
                        out_channels=out_channels[i],
                        kernel_size=kernel_sizes[i],
                        padding=((kernel_sizes[i] - 1) // 2 * dilations[i]),
                        dilation=dilations[i],
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[i] + 4),
                        reduction_ratio=(
                            (out_channels[i] // 2) if (
                                out_channels[i] > 1) else 1),
                    ),
                    nn.BatchNorm1d(out_channels[i]),
                    nn.ReLU()
                )
            )

        self.decoder = nn.ModuleList()

        self.decoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[-2],
                    kernel_size=kernel_sizes[-1],
                    padding=((kernel_sizes[-1] - 1) // 2 * dilations[-1]),
                    dilation=dilations[-1],
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[-1] + 4),
                    reduction_ratio=(
                            (out_channels[-2] // 2) if (
                                out_channels[-2] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[-2]),
                nn.ReLU()
            )
        )

        for i in range(len(out_channels) - 2, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=(2 * out_channels[i]),
                        out_channels=out_channels[i - 1],
                        kernel_size=kernel_sizes[i],
                        padding=((kernel_sizes[i] - 1) // 2 * dilations[i]),
                        dilation=dilations[i],
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[i] + 4),
                        reduction_ratio=(
                            (out_channels[i - 1] // 2) if (
                                out_channels[i - 1] > 1) else 1),
                    ),
                    nn.BatchNorm1d(out_channels[i - 1]),
                    nn.ReLU()
                )
            )

        self.decoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=(2 * out_channels[0]),
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=((kernel_sizes[0] - 1) // 2 * dilations[0]),
                    dilation=dilations[0],
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                    reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        self.classifier = nn.Sequential(
            AttendedDepthSeparableConv1d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=3,
                padding=1,
                attn=True,
                depthwise_attn_kernel_size=7,
                reduction_ratio=16
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            AttendedDepthSeparableConv1d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=3,
                padding=1,
                attn=True,
                depthwise_attn_kernel_size=7,
                reduction_ratio=16
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                out_channels[0],
                1,
                1,
                bias=False
            ),
            nn.BatchNorm1d(1)
        )

    def forward(self, sequence):
        batch_size = sequence.size()[0]

        intermediate_outs = []

        out = sequence

        for layer in self.encoder:
            out = layer(out)
            intermediate_outs.append(out)

        intermediate_outs.pop()

        for layer in self.decoder[:-1]:
            out = layer(out)
            out = torch.cat([out, intermediate_outs.pop()], dim=1)

        out = self.decoder[-1](out)
        out = self.classifier(out)
        out = torch.sigmoid(out).view(batch_size, -1)

        return out

class AtrousChannelwisePyramid(nn.Module):
    def __init__(
        self,
        in_channels=14,
        out_channels=[32, 16, 8, 4, 2],
        kernel_sizes=[9, 9, 9, 9, 9],
        dilations=[1, 2, 4, 8, 16]):
        super(AtrousChannelwisePyramid, self).__init__()
        self.backbone = nn.ModuleList()

        self.backbone.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=((kernel_sizes[0] - 1) // 2),
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                    reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        for i in range(1, len(out_channels)):
            self.backbone.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=out_channels[i - 1],
                        out_channels=out_channels[i],
                        kernel_size=kernel_sizes[i],
                        padding=((kernel_sizes[i] - 1) // 2),
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[i] + 4),
                        reduction_ratio=(
                            (out_channels[i] // 2) if (
                                out_channels[i] > 1) else 1),
                    ),
                    nn.BatchNorm1d(out_channels[i]),
                    nn.ReLU()
                )
            )

        self.pyramid = nn.ModuleList()

        self.pyramid.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[0],
                    kernel_size=1,
                    padding=0,
                    dilation=dilations[0],
                    attn=False
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        for i in range(1, len(dilations)):
            self.pyramid.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=out_channels[-1],
                        out_channels=out_channels[0],
                        kernel_size=kernel_sizes[0],
                        padding=((kernel_sizes[0] - 1) // 2 * dilations[i]),
                        dilation=dilations[i],
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                        reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1)
                    ),
                    nn.BatchNorm1d(out_channels[0]),
                    nn.ReLU()
                )
            )

        self.pyramid_compressor = nn.Sequential(
            nn.Conv1d(
                out_channels[0] * len(dilations),
                out_channels[0],
                1,
                bias=False
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.decoder = nn.ModuleList()

        self.decoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[-2],
                    kernel_size=kernel_sizes[-1],
                    padding=((kernel_sizes[-1] - 1) // 2),
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[-1] + 4),
                    reduction_ratio=(
                            (out_channels[-2] // 2) if (
                                out_channels[-2] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[-2]),
                nn.ReLU()
            )
        )

        for i in range(len(out_channels) - 2, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=(2 * out_channels[i]),
                        out_channels=out_channels[i - 1],
                        kernel_size=kernel_sizes[i],
                        padding=((kernel_sizes[i] - 1) // 2),
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[i] + 4),
                        reduction_ratio=(
                            (out_channels[i - 1] // 2) if (
                                out_channels[i - 1] > 1) else 1),
                    ),
                    nn.BatchNorm1d(out_channels[i - 1]),
                    nn.ReLU()
                )
            )

        self.decoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=(2 * out_channels[0]),
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=((kernel_sizes[0] - 1) // 2),
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                    reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        self.classifier = nn.Sequential(
            AttendedDepthSeparableConv1d(
                in_channels=(2 * out_channels[0]),
                out_channels=out_channels[0],
                kernel_size=3,
                padding=1,
                attn=True,
                depthwise_attn_kernel_size=7,
                reduction_ratio=16
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            AttendedDepthSeparableConv1d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=3,
                padding=1,
                attn=True,
                depthwise_attn_kernel_size=7,
                reduction_ratio=16
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                out_channels[0],
                1,
                1,
                bias=False
            ),
            nn.BatchNorm1d(1)
        )

    def forward(self, sequence):
        batch_size = sequence.size()[0]

        out = sequence

        skip_outs = []

        for layer in self.backbone[:-1]:
            out = layer(out)
            skip_outs.append(out)

        out = self.backbone[-1](out)

        pyramid_out = self.pyramid[0](out)

        for layer in self.pyramid[:-1]:
            pyramid_out = torch.cat([pyramid_out, layer(out)], dim=1)

        pyramid_out = self.pyramid_compressor(pyramid_out)

        for layer in self.decoder[:-1]:
            out = layer(out)
            out = torch.cat([out, skip_outs.pop()], dim=1)

        out = self.decoder[-1](out)
        out = torch.cat([pyramid_out, out], dim=1)
        out = self.classifier(out)
        out = torch.sigmoid(out).view(batch_size, -1)

        return out

class AtrousChannelwiseEncoderPyramidalDecoder(nn.Module):
    def __init__(
        self,
        in_channels=14,
        out_channels=[32, 16, 8, 4, 2],
        kernel_sizes=[9, 9, 9, 9, 9],
        bb_dilations=[1, 1, 2, 2, 3],
        pyramid_dilations=[1, 2, 6, 12, 18]):
        super(AtrousChannelwiseEncoderPyramidalDecoder, self).__init__()
        self.backbone = nn.ModuleList()

        self.backbone.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=((kernel_sizes[0] - 1) // 2 * bb_dilations[0]),
                    dilation=bb_dilations[0],
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                    reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        for i in range(1, len(out_channels)):
            self.backbone.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=out_channels[i - 1],
                        out_channels=out_channels[i],
                        kernel_size=kernel_sizes[i],
                        padding=((kernel_sizes[i] - 1) // 2 * bb_dilations[i]),
                        dilation=bb_dilations[i],
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[i] + 4),
                        reduction_ratio=(
                            (out_channels[i] // 2) if (
                                out_channels[i] > 1) else 1),
                    ),
                    nn.BatchNorm1d(out_channels[i]),
                    nn.ReLU()
                )
            )

        self.pyramid = nn.ModuleList()

        self.pyramid.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[0],
                    kernel_size=1,
                    padding=0,
                    dilation=pyramid_dilations[0],
                    attn=False
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        for i in range(1, len(pyramid_dilations)):
            self.pyramid.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=out_channels[-1],
                        out_channels=out_channels[0],
                        kernel_size=kernel_sizes[0],
                        padding=((kernel_sizes[0] - 1) // 2 * pyramid_dilations[i]),
                        dilation=pyramid_dilations[i],
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                        reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1)
                    ),
                    nn.BatchNorm1d(out_channels[0]),
                    nn.ReLU()
                )
            )

        self.pyramid_compressor = nn.Sequential(
            nn.Conv1d(
                out_channels[0] * len(pyramid_dilations),
                out_channels[0],
                1,
                bias=False
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.decoder = nn.ModuleList()

        self.decoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[-2],
                    kernel_size=kernel_sizes[-1],
                    padding=((kernel_sizes[-1] - 1) // 2),
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[-1] + 4),
                    reduction_ratio=(
                            (out_channels[-2] // 2) if (
                                out_channels[-2] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[-2]),
                nn.ReLU()
            )
        )

        for i in range(len(out_channels) - 2, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    AttendedDepthSeparableConv1d(
                        in_channels=(2 * out_channels[i]),
                        out_channels=out_channels[i - 1],
                        kernel_size=kernel_sizes[i],
                        padding=((kernel_sizes[i] - 1) // 2),
                        attn=True,
                        depthwise_attn_kernel_size=(kernel_sizes[i] + 4),
                        reduction_ratio=(
                            (out_channels[i - 1] // 2) if (
                                out_channels[i - 1] > 1) else 1),
                    ),
                    nn.BatchNorm1d(out_channels[i - 1]),
                    nn.ReLU()
                )
            )

        self.decoder.append(
            nn.Sequential(
                AttendedDepthSeparableConv1d(
                    in_channels=(2 * out_channels[0]),
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=((kernel_sizes[0] - 1) // 2),
                    attn=True,
                    depthwise_attn_kernel_size=(kernel_sizes[0] + 4),
                    reduction_ratio=(
                            (out_channels[0] // 2) if (
                                out_channels[0] > 1) else 1),
                ),
                nn.BatchNorm1d(out_channels[0]),
                nn.ReLU()
            )
        )

        self.classifier = nn.Sequential(
            AttendedDepthSeparableConv1d(
                in_channels=(2 * out_channels[0]),
                out_channels=out_channels[0],
                kernel_size=3,
                padding=1,
                attn=True,
                depthwise_attn_kernel_size=7,
                reduction_ratio=16
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.5),
            AttendedDepthSeparableConv1d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=3,
                padding=1,
                attn=True,
                depthwise_attn_kernel_size=7,
                reduction_ratio=16
            ),
            nn.BatchNorm1d(out_channels[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(
                out_channels[0],
                1,
                1,
                bias=False
            ),
            nn.BatchNorm1d(1)
        )

    def forward(self, sequence):
        batch_size = sequence.size()[0]

        out = sequence

        skip_outs = []

        for layer in self.backbone[:-1]:
            out = layer(out)
            skip_outs.append(out)

        out = self.backbone[-1](out)

        pyramid_out = self.pyramid[0](out)

        for layer in self.pyramid[:-1]:
            pyramid_out = torch.cat([pyramid_out, layer(out)], dim=1)

        pyramid_out = self.pyramid_compressor(pyramid_out)

        for layer in self.decoder[:-1]:
            out = layer(out)
            out = torch.cat([out, skip_outs.pop()], dim=1)

        out = self.decoder[-1](out)
        out = torch.cat([pyramid_out, out], dim=1)
        out = self.classifier(out)
        out = torch.sigmoid(out).view(batch_size, -1)

        return out
