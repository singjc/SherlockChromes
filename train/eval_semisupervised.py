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
    eval_by_cla=True,
    batch_size=1,
    sampling_fn=None,
    collate_fn=None
):
    # Currently only LoadingSampler returns 2 sets of idxs
    if sampling_fn:
        val_idx, test_idx = sampling_fn(
            data, test_batch_proportion)
    else:
        raise NotImplementedError

    val_set = Subset(data, val_idx, eval_by_cla)
    test_set = Subset(data, test_idx, eval_by_cla)

    if collate_fn:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=collate_fn)
    else:
        val_loader = DataLoader(
            val_set,
            batch_size=batch_size)
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size)

    return val_loader, test_loader


def eval_by_cla(model, loader, device='cpu', modulate_by_cla=True, **kwargs):
    labels_for_metrics = []
    outputs_for_metrics = []
    losses = []
    model.eval()
    orig_output_mode, model.model.output_mode = (
        model.model.output_mode, 'all')

    for batch, labels in loader:
        with torch.no_grad():
            batch = batch.to(device=device)
            labels = labels.to(device=device)
            labels_for_metrics.append(labels.cpu())
            preds = model(batch)
            strong_preds = preds['loc']
            weak_preds = preds['cla']

            if modulate_by_cla:
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

    print(
        'Eval By Cla Performance - '
        f'Accuracy: {accuracy:.4f} '
        f'Balanced accuracy: {bacc:.4f} '
        f'Precision: {precision:.4f} '
        f'Recall: {recall:.4f} '
        f'Dice: {dice:.4f} '
        f'IoU: {iou:.4f}')


def eval_by_loc(
    model,
    loader,
    device='cpu',
    modulate_by_cla=True,
    iou_threshold=0.5,
    **kwargs
):
    y_true, y_pred, y_score = [], [], []
    model.eval()
    orig_output_mode, model.model.output_mode = (
        model.model.output_mode, 'all')

    for batch, labels in loader:
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

            if modulate_by_cla:
                strong_preds = strong_preds * weak_preds

            strong_preds = strong_preds.cpu().numpy()
            weak_preds = weak_preds.cpu().numpy()
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
                label_idx = np.argwhere(
                    strong_labels[i] == 1).astype(np.int32).ravel()

                if not negative[i] and label_idx.size > 2:
                    label_left_width, label_right_width = (
                        label_idx[0], label_idx[-1])
                else:
                    negative[i] = 1

                regions_of_interest = scipy.ndimage.find_objects(
                    scipy.ndimage.label(binarized_preds[i])[0])
                overlap_found = False

                if negative[i] and not regions_of_interest:
                    # True Negative
                    y_true.append(0)
                    y_pred.append(0)
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
                        mod_right_width + 1,
                        label_left_width,
                        label_right_width + 1,
                        iou_threshold=iou_threshold
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
        f'Eval By Loc Performance at IoU Threshold {iou_threshold} - '
        f'Accuracy: {accuracy:.4f} '
        f'Avg Precision: {avg_precision:.4f} '
        f'Balanced accuracy: {bacc:.4f} '
        f'Precision: {precision:.4f} '
        f'Recall: {recall:.4f} '
        f'Dice: {dice:.4f} '
        f'IoU: {iou:.4f}')


def evaluate(
    data,
    model,
    sampling_fn=None,
    collate_fn=None,
    device='cpu',
    **kwargs
):
    if 'test_batch_proportion' not in kwargs:
        kwargs['test_batch_proportion'] = 0.1

    if 'batch_size' not in kwargs:
        kwargs['batch_size'] = 512

    val_loader_cla, test_loader_cla = get_data_loaders(
        data,
        kwargs['test_batch_proportion'],
        True,
        kwargs['batch_size'],
        sampling_fn,
        collate_fn)
    val_loader_loc, test_loader_loc = get_data_loaders(
        data,
        kwargs['test_batch_proportion'],
        False,
        kwargs['batch_size'],
        sampling_fn,
        collate_fn)

    model.to(device)

    print('Evaluating Val Data')
    print('Modulated')
    modulate_by_cla = True
    eval_by_cla(
        model, val_loader_cla, device, modulate_by_cla, **kwargs)

    for iou_threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
        eval_by_loc(
            model,
            val_loader_loc,
            device,
            modulate_by_cla,
            iou_threshold,
            **kwargs)

    print('Unmodulated')
    modulate_by_cla = False
    eval_by_cla(
        model, val_loader_cla, device, modulate_by_cla, **kwargs)

    for iou_threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
        eval_by_loc(
            model,
            val_loader_loc,
            device,
            modulate_by_cla,
            iou_threshold,
            **kwargs)

    print('Evaluating Test Data')
    print('Modulated')
    modulate_by_cla = True
    eval_by_cla(
        model, test_loader_cla, device, modulate_by_cla, **kwargs)

    for iou_threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
        eval_by_loc(
            model,
            test_loader_loc,
            device,
            modulate_by_cla,
            iou_threshold,
            **kwargs)

    print('Unmodulated')
    modulate_by_cla = False
    eval_by_cla(
        model, test_loader_cla, device, modulate_by_cla, **kwargs)

    for iou_threshold in [0.1, 0.25, 0.5, 0.75, 0.9]:
        eval_by_loc(
            model,
            test_loader_loc,
            device,
            modulate_by_cla,
            iou_threshold,
            **kwargs)
