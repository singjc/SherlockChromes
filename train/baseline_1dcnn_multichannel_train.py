import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')
sys.path.insert(0, '../optimizers')

from baseline_1dcnn_multichannel_model import BaselineChromatogramPeakDetector1DCNNMultiChannel
# from chromatograms_dataset import ChromatogramsDataset
from focal_loss import FocalLossBinary
from train import train
from transforms import ToTensor

def main(
        data,
        model,
        loss,
        collate_fn,
        optimizer_kwargs,
        train_kwargs,
        device):
    device = device
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)

    train(
        data,
        model,
        optimizer,
        loss,
        device,
        collate_fn,
        **train_kwargs)
