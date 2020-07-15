import sys
import torch
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

    if ('freeze_strong_layers' in train_kwargs
            and train_kwargs['freeze_strong_layers']):
        if 'transfer_model_path' in train_kwargs:
            model.load_state_dict(
                torch.load(train_kwargs['transfer_model_path']).state_dict(),
                strict=False
            )

        for name, param in model.named_parameters():
            if 'output_aggregator' not in name:
                param.requires_grad = False

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
