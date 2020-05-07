import sys
import torch.optim as optim

from .train_rpn import train

def main(
        data,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        train_kwargs,
        device):
    optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)

    train(
        data,
        model,
        optimizer,
        loss,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)