import sys
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')
sys.path.insert(0, '../optimizers')

from train import train

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
