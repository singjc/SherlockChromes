import importlib
import numpy as np
import os
import random
import sys
import torch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '../optimizers')

from focal_loss import FocalLossBinary

def get_data_loaders(
    data,
    test_batch_proportion=0.1,
    batch_size=1,
    u_ratio=7,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None):

    if sampling_fn:
        labeled_idx, unlabeled_idx, val_idx = sampling_fn(
            data, test_batch_proportion)
    else:
        raise NotImplementedError

    if outdir_path:
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)

        np.savetxt(
            os.path.join(outdir_path, 'labeled_idx.txt'),
            np.array(labeled_idx),
            fmt='%i'
        )
        np.savetxt(
            os.path.join(outdir_path, 'unlabeled_idx.txt'),
            np.array(unlabeled_idx),
            fmt='%i'
        )
        np.savetxt(
            os.path.join(outdir_path, 'val_idx.txt'),
            np.array(val_idx),
            fmt='%i'
        )

    labeled_set = Subset(data, labeled_idx)
    unlabeled_set = Subset(data, unlabeled_idx)
    val_set = Subset(data, val_idx)

    if collate_fn:
        labeled_loader = DataLoader(
            labeled_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=u_ratio * batch_size,
            collate_fn=collate_fn)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
    else:
        labeled_loader = DataLoader(
            labeled_set,
            batch_size=batch_size)
        unlabeled_loader = DataLoader(
            unlabeled_set,
            batch_size=u_ratio * batch_size)
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size)

    return labeled_loader, unlabeled_loader, val_loader, labeled_idx, unlabeled_idx, val_idx

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def train(
    data,
    model,
    optimizer=None,
    loss=None,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs):
    (
        labeled_loader,
        unlabeled_loader,
        val_loader,
        labeled_idx,
        unlabeled_idx,
        val_idx
    ) = get_data_loaders(
            data,
            kwargs['test_batch_proportion'],
            kwargs['batch_size'],
            kwargs['uratio'],
            sampling_fn,
            collate_fn,
            kwargs['outdir_path'])

    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters())

    if not loss:
        loss = FocalLossBinary()

    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(kwargs['transfer_model_path']).state_dict(),
            strict=False
        )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, kwargs['T_0'], T_mult=kwargs['T_mult'])

    unlabeled_loader = iter(cycle(unlabeled_loader))

    lowest_val_loss = 1

    model.to(device)

    for epoch in range(kwargs['max_epochs']):
        unlabeled_loader = iter(unlabeled_loader)
        if not ('scheduler_step_on_iter' in kwargs and
                    kwargs['scheduler_step_on_iter']):
                scheduler.step()

        iters, avg_loss = 0, 0

        for labeled_batch, labels in labeled_loader:
            unlabeled_batch, _ = next(unlabeled_loader)
            labeled_batch = labeled_batch.to(device=device)
            labels = labels.to(device=device)
            unlabeled_batch = unlabeled_batch.to(device=device)

            model.train()

            if ('scheduler_step_on_iter' in kwargs and
                    kwargs['scheduler_step_on_iter']):
                scheduler.step()
                
            optimizer.zero_grad()

            loss_out = model(unlabeled_batch, labeled_batch, labels)
            loss_out.backward()
            optimizer.step()
            iters+= 1
            iter_loss = loss_out.item()
            avg_loss+= iter_loss
            print(f'Training - Iter: {iters} Iter loss: {iter_loss:.8f}')

        print(f'Training - Epoch: {epoch} Avg loss: {(avg_loss / iters):.8f}')

        iters, avg_loss = 0, 0
        for batch, labels in val_loader:
            model.eval()
            with torch.no_grad():
                batch = batch.to(device=device)
                labels = labels.to(device=device)
                preds = model(batch)
                loss_out = loss(preds, labels)
                iters+= 1
                avg_loss+= loss_out.item()

        avg_loss/= iters

        if avg_loss <= lowest_val_loss:
            save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_loss={avg_loss}.pth"
            )

            if 'save_whole' in kwargs and kwargs['save_whole']:
                torch.save(model, save_path)
            else:
                torch.save(model.state_dict(), save_path)

        print(f'Validation - Epoch: {epoch} Avg loss: {(avg_loss):.8f}')
