import sys
import torch.optim as optim

from train.train_weakly_supervised import train

def main(
        data,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        scheduler_kwargs,
        train_kwargs,
        device):
    optimizer = train_kwargs.pop('optimizer', None)

    if optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    elif optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    scheduler = train_kwargs.pop('scheduler', None)

    if scheduler == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, **scheduler_kwargs)
    elif scheduler == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **scheduler_kwargs)

    train(
        data,
        model,
        optimizer,
        scheduler,
        loss,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)
