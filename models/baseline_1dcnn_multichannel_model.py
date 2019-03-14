import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineChromatogramPeakDetector1DCNNMultiChannel(nn.Module):
    def __init__(
        self,
        batch_size,
        in_channels=6,
        out_channels=[128, 128, 64, 128, 128, 64],
        kernel_size=3,
        strides=[1, 1, 2, 1, 1, 2],
        paddings=[1, 1, 1, 1, 1, 1]
        ):
        super(
            BaselineChromatogramPeakDetector1DCNNMultiChannel,
            self).__init__()

        self.batch_size = batch_size

        self.bn = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels[0],
            kernel_size,
            stride=strides[0],
            padding=paddings[0])

        self.conv2 = nn.Conv1d(
            out_channels[0],
            out_channels[1],
            kernel_size,
            stride=strides[1],
            padding=paddings[1])

        self.conv3 = nn.Conv1d(
            out_channels[1],
            out_channels[2],
            kernel_size,
            stride=strides[2],
            padding=paddings[2])
        
        self.conv4 = nn.Conv1d(
            out_channels[2],
            out_channels[3],
            kernel_size,
            stride=strides[3],
            padding=paddings[3])
        
        self.conv5 = nn.Conv1d(
            out_channels[3],
            out_channels[4],
            kernel_size,
            stride=strides[4],
            padding=paddings[4])

        self.conv6 = nn.Conv1d(
            out_channels[4],
            out_channels[5],
            kernel_size,
            stride=strides[5],
            padding=paddings[5])

        self.out_conv1 = nn.Conv1d(
            out_channels[5],
            32,
            kernel_size=20,
            stride=1,
            padding=0)

        self.out_conv2 = nn.Conv1d(
            32,
            16,
            kernel_size=1,
            stride=1,
            padding=0)
        
        self.out_conv3 = nn.Conv1d(
            16,
            1,
            kernel_size=1,
            stride=1,
            padding=0)
        
    def forward(self, chromatogram):
        out = self.bn(F.relu(self.conv1(chromatogram)))

        out = F.relu(self.conv2(out))

        out = F.relu(self.conv3(out))

        out = F.relu(self.conv4(out))

        out = F.relu(self.conv5(out))

        out = F.relu(self.conv6(out))

        out = F.relu(self.out_conv1(out))

        out = F.relu(self.out_conv2(out))

        out = torch.sigmoid(self.out_conv3(out)).reshape(
            self.batch_size)

        return out
