import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')
sys.path.insert(0, '../optimizers')

from baseline_lstm_model import BaselineChromatogramPeakDetectorLSTM
from chromatograms_dataset import ChromatogramsDataset
from focal_loss import FocalLossBinary
from train import train
from transforms import ToTensor
from visualizer import plot_chromatogram, plot_confusion_matrix

def main(data_kwargs, model_kwargs, loss_kwargs, optimizer_kwargs, train_kwargs, device):
    device = device
    data = ChromatogramsDataset(**data_kwargs, transform=ToTensor())
    model = BaselineChromatogramPeakDetectorLSTM(**model_kwargs, device=device)
    loss_function = FocalLossBinary(**loss_kwargs)
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)

    train(
        data,
        model,
        optimizer,
        loss_function,
        device,
        **train_kwargs)
