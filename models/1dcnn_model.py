import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        intermediate_channels=32):
        super(DepthSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels * intermediate_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.pointwise = nn.Conv1d(
            in_channels * intermediate_channels,
            out_channels,
            1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class ChromatogramPeakDetectorDepthSeparable1DCNN(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels=6,
        out_channels=[64, 64, 64, 1],
        kernel_sizes=[3, 1, 1, 20],
        strides=[1, 1, 1, 1],
        paddings=[1, 0, 0, 0],
        intermediate_channels=[128]):
        super(
            ChromatogramPeakDetectorDepthSeparable1DCNN,
            self).__init__()

        self.batch_size = batch_size

        self.in1 = nn.InstanceNorm1d(in_channels)

        self.depthseparable1 = DepthSeparableConv1d(
            in_channels,
            out_channels[0],
            kernel_sizes[0],
            stride=strides[0],
            padding=paddings[0],
            intermediate_channels=intermediate_channels[0])

        self.bn1 = nn.BatchNorm1d(out_channels[0])

        self.cross_channel = nn.Sequential(
            nn.Conv1d(
                out_channels[0],
                out_channels[1],
                kernel_sizes[1],
                stride=strides[1],
                padding=paddings[1]),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels[1]),
            nn.Conv1d(
                out_channels[1],
                out_channels[2],
                kernel_sizes[2],
                stride=strides[2],
                padding=paddings[2]),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels[2]),
            nn.Conv1d(
                out_channels[2],
                out_channels[3],
                kernel_sizes[3],
                stride=strides[3],
                padding=paddings[3]))
    
    def forward(self, chromatogram):
        out = self.in1(chromatogram)
        out = self.bn1(F.relu(self.depthseparable1(out)))
        out = self.cross_channel(out)
        out = torch.sigmoid(out).reshape(self.batch_size)

        return out

class ChromatogramPeakDetector1DCNNMultichannel(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels=6,
        out_channels=[32, 64, 128, 64, 32, 16, 128, 1],
        kernel_sizes=[3, 3, 3, 1, 1, 1, 10, 1],
        strides=[1, 1, 1, 1, 1, 1, 1, 1],
        paddings=[1, 1, 1, 0, 0, 0, 0, 0]
        ):
        super(
            ChromatogramPeakDetector1DCNNMultichannel,
            self).__init__()

        self.batch_size = batch_size

        self.in1 = nn.InstanceNorm1d(in_channels)

        self.bn1 = nn.BatchNorm1d(out_channels[0])
        self.bn2 = nn.BatchNorm1d(out_channels[1])
        self.bn3 = nn.BatchNorm1d(out_channels[2])
        self.bn4 = nn.BatchNorm1d(out_channels[3])
        self.bn5 = nn.BatchNorm1d(out_channels[4])
        self.bn6 = nn.BatchNorm1d(out_channels[5])
        self.bn7 = nn.BatchNorm1d(out_channels[6])

        self.maxpool = nn.MaxPool1d(2)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels[0],
            kernel_sizes[0],
            stride=strides[0],
            padding=paddings[0])

        self.conv2 = nn.Conv1d(
            out_channels[0],
            out_channels[1],
            kernel_sizes[1],
            stride=strides[1],
            padding=paddings[1])

        self.conv3 = nn.Conv1d(
            out_channels[1],
            out_channels[2],
            kernel_sizes[2],
            stride=strides[2],
            padding=paddings[2])

        self.conv4 = nn.Conv1d(
            out_channels[2],
            out_channels[3],
            kernel_sizes[3],
            stride=strides[3],
            padding=paddings[3])

        self.conv5 = nn.Conv1d(
            out_channels[3],
            out_channels[4],
            kernel_sizes[4],
            stride=strides[4],
            padding=paddings[4])
        
        self.conv6 = nn.Conv1d(
            out_channels[4],
            out_channels[5],
            kernel_sizes[5],
            stride=strides[5],
            padding=paddings[5])

        self.out_conv1 = nn.Conv1d(
            out_channels[5],
            out_channels[6],
            kernel_sizes[6],
            stride=strides[6],
            padding=paddings[6])

        self.out_conv2 = nn.Conv1d(
            out_channels[6],
            out_channels[7],
            kernel_sizes[7],
            stride=strides[7],
            padding=paddings[7])
        
    def forward(self, chromatogram):
        chromatogram = self.in1(chromatogram)

        out = self.bn1(F.relu(self.conv1(chromatogram)))

        out = self.bn2(F.relu(self.conv2(out)))

        out = self.maxpool(out)

        out = self.bn3(F.relu(self.conv3(out)))

        out = self.bn4(F.relu(self.conv4(out)))

        out = self.bn5(F.relu(self.conv5(out)))

        out = self.bn6(F.relu(self.conv6(out)))

        out = self.bn7(F.relu(self.out_conv1(out)))

        out = self.out_conv2(out)

        out = torch.sigmoid(out).reshape(self.batch_size)

        return out

class ChromatogramPeakDetector1DCNNMultihead(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels=6,
        out_channels=[32, 32, 16, 100, 1],
        kernel_sizes=[3, 3, 3, 5, 1],
        strides=[1, 1, 1, 1, 1],
        paddings=[1, 1, 1, 0, 0]
        ):
        super(
            ChromatogramPeakDetector1DCNNMultihead,
            self).__init__()

        self.batch_size = batch_size

        self.maxpool = nn.MaxPool1d(2)

        self.heads = nn.ModuleList()

        for channel in range(in_channels):
            self.heads.append(
                nn.Sequential(
                    nn.Conv1d(
                        1,
                        out_channels[0],
                        kernel_sizes[0],
                        stride=strides[0],
                        padding=paddings[0]),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels[0]),
                    nn.Conv1d(
                        out_channels[0],
                        out_channels[1],
                        kernel_sizes[1],
                        stride=strides[1],
                        padding=paddings[0]),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels[1])))

        self.combined = nn.Sequential(
            nn.Conv1d(
                in_channels * out_channels[1],
                out_channels[2],
                kernel_sizes[2],
                stride=strides[2],
                padding=paddings[2]),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels[2]),
            self.maxpool,
            nn.Conv1d(
                out_channels[2],
                out_channels[3],
                kernel_sizes[3],
                stride=strides[3],
                padding=paddings[3]),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels[3]),
            nn.Conv1d(
                out_channels[3],
                out_channels[4],
                kernel_sizes[4],
                stride=strides[4],
                padding=paddings[4]))
        
    def forward(self, chromatogram):
        out = torch.cat(
            [self.heads[i](
                    chromatogram[:, i, :].reshape(
                        chromatogram.size(0),
                        1,
                        chromatogram.size(2))) for i in range(
                            chromatogram.size(1))], 1)

        out = self.combined(out)

        out = torch.sigmoid(out).reshape(self.batch_size)

        return out