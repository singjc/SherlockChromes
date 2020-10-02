import importlib
import numpy as np
import os
import random
import scipy.ndimage
import sys
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from datasets.chromatograms_dataset import Subset
from optimizers.focal_loss import FocalLossBinary
from utils.general_utils import overlaps


def get_data_loaders(
    data,
    test_batch_proportion=0.1,
    use_weak_labels=False,
    eval_by_cla=True,
    batch_size=1,
    u_ratio=7,
    sampling_fn=None,
    collate_fn=None,
    outdir_path=None
):
    # Currently only LoadingSampler returns 4 sets of idxs
    if sampling_fn:
        labeled_idx, unlabeled_idx, val_idx, test_idx = sampling_fn(
            data, test_batch_proportion)
    else:
        raise NotImplementedError

    if outdir_path:
        if not os.path.isdir(outdir_path):
            os.mkdir(outdir_path)

        np.savetxt(
            os.path.join(outdir_path, 'labeled_idx.txt'),
            np.array(labeled_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'unlabeled_idx.txt'),
            np.array(unlabeled_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'val_idx.txt'),
            np.array(val_idx),
            fmt='%i')
        np.savetxt(
            os.path.join(outdir_path, 'test_idx.txt'),
            np.array(test_idx),
            fmt='%i')

    labeled_set = Subset(data, labeled_idx, use_weak_labels)
    unlabeled_set = Subset(data, unlabeled_idx, False)
    val_set = Subset(data, val_idx, eval_by_cla)
    test_set = Subset(data, test_idx, False)

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
        test_loader = DataLoader(
            test_set,
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
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size)

    return labeled_loader, unlabeled_loader, val_loader, test_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train(
    data,
    model,
    optimizer=None,
    scheduler=None,
    loss=None,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs
):
    (
        labeled_loader,
        unlabeled_loader,
        val_loader,
        test_loader
    ) = get_data_loaders(
            data,
            kwargs['test_batch_proportion'],
            kwargs['use_weak_labels'],
            kwargs['eval_by_cla'],
            kwargs['batch_size'],
            kwargs['uratio'],
            sampling_fn,
            collate_fn,
            kwargs['outdir_path'])

    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters())

    if not loss:
        loss = FocalLossBinary()

    if not scheduler:
        scheduler = CosineAnnealingWarmRestarts(optimizer, 10)

    if 'transfer_model_path' in kwargs:
        model.load_state_dict(
            torch.load(kwargs['transfer_model_path']).state_dict(),
            strict=False)

    unlabeled_loader = iter(cycle(unlabeled_loader))

    highest_bacc, highest_dice, highest_iou, lowest_loss = 0, 0, 0, 100

    model.to(device)

    num_batches = len(labeled_loader)

    for epoch in range(kwargs['max_epochs']):
        iters, avg_loss = 0, 0
        model.train()

        for i, sample in enumerate(labeled_loader):
            labeled_batch, labels = sample
            unlabeled_batch, _ = next(unlabeled_loader)
            labeled_batch = labeled_batch.to(device=device)
            labels = labels.to(device=device)
            unlabeled_batch = unlabeled_batch.to(device=device)
            optimizer.zero_grad()
            loss_out = model(unlabeled_batch, labeled_batch, labels)
            loss_out.backward()
            optimizer.step()
            scheduler.step(epoch + i / num_batches)
            iters += 1
            iter_loss = loss_out.item()
            avg_loss += iter_loss

            print(f'Training - Iter: {iters} Iter loss: {iter_loss:.8f}')

        if not ('scheduler_step_on_iter' in kwargs and
                kwargs['scheduler_step_on_iter']):
            scheduler.step()

        print(f'Training - Epoch: {epoch} Avg loss: {(avg_loss / iters):.8f}')

        labels_for_metrics = []
        outputs_for_metrics = []
        losses = []
        model.eval()
        orig_output_mode, model.model.output_mode = (
            model.model.output_mode, 'all')

        for batch, labels in val_loader:
            with torch.no_grad():
                batch = batch.to(device=device)
                labels = labels.to(device=device)
                labels_for_metrics.append(labels.cpu())
                preds = model(batch)
                strong_preds = preds['loc']
                weak_preds = preds['cla']

                if (
                    kwargs['use_weak_labels']
                    or kwargs['enforce_weak_consistency']
                ):
                    b, _ = strong_preds.size()
                    strong_preds = strong_preds * weak_preds

                binarized_preds = np.where(strong_preds.cpu() >= 0.5, 1, 0)
                inverse_binarized_preds = (1 - binarized_preds)
                global_preds = np.zeros(labels.shape)

                for i in range(len(strong_preds)):
                    gaps = scipy.ndimage.find_objects(
                        scipy.ndimage.label(inverse_binarized_preds[i])[0])

                    for gap in gaps:
                        gap = gap[0]
                        gap_length = gap.stop - gap.start

                        if gap_length < 3:
                            binarized_preds[i][gap.start:gap.stop] = 1

                    regions_of_interest = scipy.ndimage.find_objects(
                        scipy.ndimage.label(binarized_preds[i])[0])

                    for roi in regions_of_interest:
                        roi = roi[0]
                        roi_length = roi.stop - roi.start

                        if 2 < roi_length < 60:
                            global_preds[i] = 1
                            break

                outputs_for_metrics.append(global_preds)
                loss_out = loss(
                    torch.from_numpy(global_preds).to(device=device).float(),
                    labels)
                losses.append(loss_out.item())

        model.model.output_mode = orig_output_mode
        labels_for_metrics = np.concatenate(labels_for_metrics, axis=0)
        outputs_for_metrics = np.concatenate(outputs_for_metrics, axis=0)
        accuracy = accuracy_score(labels_for_metrics, outputs_for_metrics)
        bacc = balanced_accuracy_score(
            labels_for_metrics, outputs_for_metrics)
        precision = precision_score(labels_for_metrics, outputs_for_metrics)
        recall = recall_score(labels_for_metrics, outputs_for_metrics)
        dice = f1_score(labels_for_metrics, outputs_for_metrics)
        iou = jaccard_score(labels_for_metrics, outputs_for_metrics)
        avg_loss = np.mean(losses)

        print(
            f'Validation - Epoch: {epoch} '
            f'Accuracy: {accuracy:.8f} '
            f'Balanced accuracy: {bacc:.8f} '
            f'Precision: {precision:.8f} '
            f'Recall: {recall:.8f} '
            f'Dice: {dice:.8f} '
            f'IoU: {iou:.8f} '
            f'Avg loss: {avg_loss:.8f} ')

        save_path = ''

        if bacc > highest_bacc:
            save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_bacc={bacc}.pth")

            highest_bacc = bacc

            if dice > highest_dice:
                highest_dice = dice

            if iou > highest_iou:
                highest_iou = iou

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
        elif dice > highest_dice:
            save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_dice={dice}.pth")

            highest_dice = dice

            if iou > highest_iou:
                highest_iou = iou

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
        elif iou > highest_iou:
            save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_iou={iou}.pth")
            highest_iou = iou

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
        elif avg_loss < lowest_loss:
            save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_loss={avg_loss}"
                '.pth')
            lowest_loss = avg_loss

        if save_path:
            if 'save_whole' in kwargs and kwargs['save_whole']:
                torch.save(model, save_path)
            else:
                torch.save(model.state_dict(), save_path)

        y_true, y_pred, y_score = [], [], []
        model.eval()
        orig_output_mode, model.model.output_mode = (
            model.model.output_mode, 'all')

        for batch, labels in test_loader:
            with torch.no_grad():
                batch = batch.to(device=device)
                strong_labels = labels.to(device=device)
                weak_labels = torch.max(
                    strong_labels, dim=1, keepdim=True)[0].cpu().numpy()
                strong_labels = strong_labels.cpu().numpy()
                negative = 1 - weak_labels
                preds = model(batch)
                strong_preds = preds['loc']
                weak_preds = preds['cla']

                if (
                    kwargs['use_weak_labels']
                    or kwargs['enforce_weak_consistency']
                ):
                    b, _ = strong_preds.size()
                    strong_preds = strong_preds * weak_preds

                strong_preds = strong_preds.cpu().numpy()
                weak_preds = weak_preds.cpu().numpy()
                label_idx = np.argwhere(
                    strong_labels == 1).astype(np.int32)
                label_idx = np.split(
                    label_idx[:, 1],
                    np.unique(label_idx[:, 0], return_index=True)[1])[1:]
                binarized_preds = np.where(
                    strong_preds >= 0.5, 1, 0).astype(np.int32)
                inverse_binarized_preds = (1 - binarized_preds)

                for i in range(len(strong_preds)):
                    gaps = scipy.ndimage.find_objects(
                        scipy.ndimage.label(inverse_binarized_preds[i])[0])

                    for gap in gaps:
                        gap = gap[0]
                        gap_length = gap.stop - gap.start

                        if gap_length < 3:
                            binarized_preds[i][gap.start:gap.stop] = 1

                    label_left_width, label_right_width = None, None

                    if not negative[i] and label_idx[i]:
                        label_left_width, label_right_width = (
                            label_idx[i][0], label_idx[i][-1])
                    else:
                        negative[i] = 1

                    regions_of_interest = scipy.ndimage.find_objects(
                        scipy.ndimage.label(binarized_preds[i])[0])
                    overlap_found = False

                    if negative[i] and not regions_of_interest:
                        # True Negative
                        y_true.append(0)
                        y_pred.append(0)

                        if (
                            kwargs['use_weak_labels']
                            or kwargs['enforce_weak_consistency']
                        ):
                            y_score.append(weak_preds[i][0])
                        else:
                            y_score.append(np.max(strong_preds[i]))

                    for j in range(len(regions_of_interest)):
                        mod_left_width, mod_right_width = None, None
                        region = regions_of_interest[j]
                        score = np.sum(strong_preds[i][region])
                        region = region[0]
                        start_idx, end_idx = region.start, region.stop
                        mod_left_width, mod_right_width = (
                            region.start, region.stop - 1)
                        score = score / (region.stop - region.start)

                        if negative[i]:
                            # False Positive
                            y_true.append(0)
                        elif not overlaps(
                            mod_left_width,
                            mod_right_width,
                            label_left_width,
                            label_right_width,
                            threshold=0.33
                        ):
                            # False Positive
                            y_true.append(0)
                        else:
                            # True Positive
                            y_true.append(1)
                            overlap_found = True

                        y_pred.append(1)
                        y_score.append(score)

                    if not negative[i] and not overlap_found:
                        # False Negative
                        label_region_score = np.sum(
                            strong_preds[i][
                                label_left_width:label_right_width + 1])
                        label_region_score = (
                            label_region_score
                            / label_right_width + 1 - label_left_width)
                        y_true.append(1)
                        y_pred.append(0)
                        y_score.append(label_region_score)

        model.model.output_mode = orig_output_mode
        y_true, y_pred, y_score = (
            np.array(y_true, dtype=np.int32),
            np.array(y_pred, dtype=np.int32),
            np.array(y_score, dtype=np.int32))
        accuracy = accuracy_score(y_true, y_pred)
        avg_precision = average_precision_score(y_true, y_score)
        bacc = balanced_accuracy_score(
            y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        dice = f1_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred)

        print(
            f'Test - Epoch: {epoch} '
            f'Accuracy: {accuracy:.4f} '
            f'Avg Precision: {avg_precision} '
            f'Balanced accuracy: {bacc:.4f} '
            f'Precision: {precision:.4f} '
            f'Recall: {recall:.4f} '
            f'Dice: {dice:.4f} '
            f'IoU: {iou:.4f}')

    save_path = save_path = os.path.join(
                kwargs['outdir_path'],
                f"{kwargs['model_savename']}_model_{epoch}_final.pth")

    if 'save_whole' in kwargs and kwargs['save_whole']:
        torch.save(model, save_path)
    else:
        torch.save(model.state_dict(), save_path)
