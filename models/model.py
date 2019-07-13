import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.insert(0, '../models')

from custom_layers_and_blocks import DepthSeparableConv1d, GlobalContextBlock1d

class ChromatogramPeakDetectorAtrousEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels=14,
        out_channels=[32, 16, 8, 4, 2],
        kernel_sizes=[3, 3, 3, 3, 3],
        paddings=[1, 1, 2, 2, 3], 
        dilations=[1, 1, 2, 2, 3]):
        super(ChromatogramPeakDetectorAtrousEncoderDecoder, self).__init__()
        self.encoder = nn.ModuleList()

        self.encoder.append(
            nn.Sequential(
                DepthSeparableConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=paddings[0],
                    dilation=dilations[0]
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels[0]),
                GlobalContextBlock1d(
                    in_channels=out_channels[0],
                    reduction_ratio=(out_channels[0] // 2)
                )
            )
        )

        for i in range(1, len(out_channels)):
            self.encoder.append(
                nn.Sequential(
                    DepthSeparableConv1d(
                        in_channels=out_channels[i - 1],
                        out_channels=out_channels[i],
                        kernel_size=kernel_sizes[i],
                        padding=paddings[i],
                        dilation=dilations[i]
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels[i]),
                    GlobalContextBlock1d(
                        in_channels=out_channels[i],
                        reduction_ratio=(out_channels[i] // 2)
                    )
                )
            )

        self.decoder = nn.ModuleList()

        self.decoder.append(
            nn.Sequential(
                DepthSeparableConv1d(
                    in_channels=out_channels[-1],
                    out_channels=out_channels[-2],
                    kernel_size=kernel_sizes[-1],
                    padding=paddings[-1],
                    dilation=dilations[-1]
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels[-2]),
                GlobalContextBlock1d(
                    in_channels=out_channels[-2],
                    reduction_ratio=(out_channels[-1] // 2)
                )
            )
        )

        for i in range(len(out_channels) - 2, 0, -1):
            self.decoder.append(
                nn.Sequential(
                    DepthSeparableConv1d(
                        in_channels=(2 * out_channels[i]),
                        out_channels=out_channels[i - 1],
                        kernel_size=kernel_sizes[i],
                        padding=paddings[i],
                        dilation=dilations[i]
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels[i - 1]),
                    GlobalContextBlock1d(
                        in_channels=out_channels[i - 1],
                        reduction_ratio=(out_channels[i] // 2)
                    )
                )
            )

        self.decoder.append(
            nn.Sequential(
                DepthSeparableConv1d(
                    in_channels=(2 * out_channels[0]),
                    out_channels=out_channels[0],
                    kernel_size=kernel_sizes[0],
                    padding=paddings[0],
                    dilation=dilations[0]
                ),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels[0]),
                GlobalContextBlock1d(
                    in_channels=out_channels[0],
                    reduction_ratio=(out_channels[0] // 2)
                )
            )
        )

        self.classifier = nn.Sequential(
                DepthSeparableConv1d(
                    in_channels=out_channels[0],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    bias=False,
                    depth_multiplier=1,
                    intermediate_nonlinearity=False
                ),
                nn.BatchNorm1d(1)
        )

    def forward(self, chromatogram):
        batch_size = chromatogram.size()[0]

        intermediate_outs = []

        out = chromatogram

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
