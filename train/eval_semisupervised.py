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
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .train_semisupervised import cycle
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


def eval_by_cla(
    model,
    loader,
    strong_label_loader,
    device='cpu',
    modulate_by_cla=True,
    **kwargs
):
    labels_for_metrics = []
    outputs_for_metrics = []
    scores_for_metrics = []
    losses = []
    model.eval()
    orig_output_mode, model.model.output_mode = (
        model.model.output_mode, 'all')

    for batch, labels in loader:
        with torch.no_grad():
            batch = batch.to(device=device)
            labels = labels.to(device=device)
            _, strong_labels = next(strong_label_loader)
            derived_weak_labels = torch.max(
                strong_labels, dim=1)[0].cpu().numpy().ravel()
            labels = (
                labels.cpu().numpy().ravel() * derived_weak_labels)
            labels_for_metrics.append(labels)
            preds = model(batch)
            strong_preds = preds['loc'].cpu().numpy()
            weak_preds = preds['cla'].cpu().numpy()
            scores_for_metrics.append(weak_preds)

            if modulate_by_cla:
                strong_preds = strong_preds * weak_preds

            binarized_preds = np.where(
                strong_preds >= kwargs['output_threshold'], 1, 0)
            inverse_binarized_preds = (1 - binarized_preds)
            global_preds = np.zeros(labels.shape)

            for i in range(len(strong_preds)):
                if 'fill_gaps' in kwargs and kwargs['fill_gaps']:
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

                    if 3 <= roi_length <= 36:
                        global_preds[i] = 1
                        break

                if 'be_generous' in kwargs and kwargs['be_generous']:
                    if np.max(strong_preds[i]) >= kwargs['output_threshold']:
                        global_preds[i] = 1

            outputs_for_metrics.append(global_preds)

    model.model.output_mode = orig_output_mode
    labels_for_metrics = np.concatenate(labels_for_metrics, axis=0)
    outputs_for_metrics = np.concatenate(outputs_for_metrics, axis=0)
    scores_for_metrics = np.concatenate(scores_for_metrics, axis=0)
    accuracy = accuracy_score(labels_for_metrics, outputs_for_metrics)
    avg_precision = average_precision_score(
        labels_for_metrics, scores_for_metrics)
    bacc = balanced_accuracy_score(
        labels_for_metrics, outputs_for_metrics)
    precision = precision_score(labels_for_metrics, outputs_for_metrics)
    recall = recall_score(labels_for_metrics, outputs_for_metrics)
    dice = f1_score(labels_for_metrics, outputs_for_metrics)
    iou = jaccard_score(labels_for_metrics, outputs_for_metrics)
    tn, fp, fn, tp = confusion_matrix(
        labels_for_metrics, outputs_for_metrics).ravel()

    print(
        'Eval By Cla Performance - '
        f'Accuracy: {accuracy:.4f} '
        f'Avg precision: {avg_precision:.4f} '
        f'Balanced accuracy: {bacc:.4f} '
        f'Precision: {precision:.4f} '
        f'Recall: {recall:.4f} '
        f'Dice: {dice:.4f} '
        f'IoU: {iou:.4f} '
        f'TN/FP/FN/TP: {tn}/{fp}/{fn}/{tp}')


def eval_by_loc(
    model,
    loader,
    weak_label_loader,
    device='cpu',
    modulate_by_cla=True,
    **kwargs
):
    y_true, y_pred, y_score = [], [], []
    model.eval()
    orig_output_mode, model.model.output_mode = (
        model.model.output_mode, 'all')
    txt_line_num = 0
    false_positive_line_nums = []
    false_negative_line_nums = []

    for batch, labels in loader:
        with torch.no_grad():
            batch = batch.to(device=device)
            strong_labels = labels.to(device=device)
            _, weak_labels = next(weak_label_loader)
            derived_weak_labels = torch.max(
                strong_labels, dim=1)[0].cpu().numpy().ravel()
            weak_labels = (
                weak_labels.cpu().numpy().ravel() * derived_weak_labels)
            strong_labels = strong_labels.cpu().numpy()
            negative = 1 - weak_labels
            preds = model(batch)
            strong_preds = preds['loc'].cpu().numpy()
            weak_preds = preds['cla'].cpu().numpy()

            if modulate_by_cla:
                strong_preds = strong_preds * weak_preds

            binarized_preds = np.where(
                strong_preds >= kwargs['output_threshold'], 1, 0).astype(
                    np.int32)
            inverse_binarized_preds = (1 - binarized_preds)

            for i in range(len(strong_preds)):
                txt_line_num += 1
                if 'fill_gaps' in kwargs and kwargs['fill_gaps']:
                    gaps = scipy.ndimage.find_objects(
                        scipy.ndimage.label(inverse_binarized_preds[i])[0])

                    for gap in gaps:
                        gap = gap[0]
                        gap_length = gap.stop - gap.start

                        if gap_length < 3:
                            binarized_preds[i][gap.start:gap.stop] = 1

                label_left_width, label_right_width = None, None

                if not negative[i]:
                    label_idx = np.argwhere(
                        strong_labels[i] == 1).astype(np.int32).ravel()
                    label_left_width, label_right_width = (
                        label_idx[0], label_idx[-1])

                regions_of_interest = scipy.ndimage.find_objects(
                    scipy.ndimage.label(binarized_preds[i])[0])
                min_length = 3

                if 'be_generous' in kwargs and kwargs['be_generous']:
                    min_length = 1

                regions_of_interest = [
                    roi[0] for roi in regions_of_interest
                    if min_length <= roi[0].stop - roi[0].start <= 36]
                overlap_found = False

                if negative[i] and not regions_of_interest:
                    # True Negative
                    y_true.append(0)
                    y_pred.append(0)
                    y_score.append(0)

                if (
                    'top_choice' in kwargs
                    and kwargs['top_choice']
                    and regions_of_interest
                ):
                    scores = [
                        np.max(strong_preds[i][roi.start:roi.stop])
                        for roi in regions_of_interest]
                    best_region_idx = np.argmax(scores)
                    score = scores[best_region_idx]
                    best_region = regions_of_interest[best_region_idx]

                    if negative[i] or not overlaps(
                        best_region.start,
                        best_region.stop,
                        label_left_width,
                        label_right_width + 1,
                        iou_threshold=0.5
                    ):
                        # False Positive
                        false_positive_line_nums.append(txt_line_num)
                        y_true.append(0)
                    else:
                        # True Positive
                        y_true.append(1)
                        overlap_found = True

                    y_pred.append(1)
                    y_score.append(score)
                    regions_of_interest = []

                for roi in regions_of_interest:
                    score = np.max(strong_preds[i][roi.start:roi.stop])

                    if negative[i] or not overlaps(
                        roi.start,
                        roi.stop,
                        label_left_width,
                        label_right_width + 1,
                        iou_threshold=0.5
                    ):
                        # False Positive
                        false_positive_line_nums.append(txt_line_num)
                        y_true.append(0)
                    else:
                        # True Positive
                        y_true.append(1)
                        overlap_found = True

                    y_pred.append(1)
                    y_score.append(score)

                if not negative[i] and not overlap_found:
                    false_negative_line_nums.append(txt_line_num)
                    # False Negative
                    label_region_score = np.max(
                        strong_preds[i][
                            label_left_width:label_right_width + 1])
                    y_true.append(1)
                    y_pred.append(0)
                    y_score.append(label_region_score)

    model.model.output_mode = orig_output_mode
    y_true, y_pred, y_score = (
        np.array(y_true, dtype=np.int32),
        np.array(y_pred, dtype=np.int32),
        np.array(y_score, dtype=np.float32))
    accuracy = accuracy_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_score)
    bacc = balanced_accuracy_score(
        y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    dice = f1_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(
        f'Eval By Loc Performance - '
        f'Accuracy: {accuracy:.4f} '
        f'Avg precision: {avg_precision:.4f} '
        f'Balanced accuracy: {bacc:.4f} '
        f'Precision: {precision:.4f} '
        f'Recall: {recall:.4f} '
        f'Dice: {dice:.4f} '
        f'IoU: {iou:.4f} '
        f'TN/FP/FN/TP: {tn}/{fp}/{fn}/{tp}')

    if 'print_failures' in kwargs and kwargs['print_failures']:
        print(set(false_positive_line_nums))
        print(false_negative_line_nums)

    if 'plot_pr' in kwargs and kwargs['plot_pr']:
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        colors = iter(cm.rainbow(np.linspace(0, 1, 1)))
        labels = [f'Input Network']

        lines = []
        precision, recall, threshold = precision_recall_curve(
            y_true, y_score)

        l, = plt.plot(recall, precision, color=next(colors), lw=2)
        lines.append(l)
        labels[0] += f', AP: {avg_precision}'

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks([i*0.05 for i in range(0, 21)])
        plt.yticks([i*0.05 for i in range(0, 21)])
        plt.grid()
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

        plt.show()

        print(y_true.tolist())
        print(y_score.tolist())


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

    if 'output_threshold' not in kwargs:
        kwargs['output_threshold'] = 0.5

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
    val_loader_cla_sl, test_loader_cla_sl = get_data_loaders(
        data,
        kwargs['test_batch_proportion'],
        False,
        kwargs['batch_size'],
        sampling_fn,
        collate_fn)
    val_loader_loc_wl, test_loader_loc_wl = get_data_loaders(
        data,
        kwargs['test_batch_proportion'],
        True,
        kwargs['batch_size'],
        sampling_fn,
        collate_fn)
    val_loader_cla_sl, test_loader_cla_sl = (
        iter(cycle(val_loader_cla_sl)), iter(cycle(test_loader_cla_sl)))
    val_loader_loc_wl, test_loader_loc_wl = (
        iter(cycle(val_loader_loc_wl)), iter(cycle(test_loader_loc_wl)))

    model.to(device)

    print('Evaluating Val Data')
    print('Modulated')
    modulate_by_cla = True
    eval_by_cla(
        model,
        val_loader_cla,
        val_loader_cla_sl,
        device,
        modulate_by_cla,
        **kwargs)

    eval_by_loc(
        model,
        val_loader_loc,
        val_loader_loc_wl,
        device,
        modulate_by_cla,
        **kwargs)

    print('Unmodulated')
    modulate_by_cla = False
    eval_by_cla(
        model,
        val_loader_cla,
        val_loader_cla_sl,
        device,
        modulate_by_cla,
        **kwargs)

    eval_by_loc(
        model,
        val_loader_loc,
        val_loader_loc_wl,
        device,
        modulate_by_cla,
        **kwargs)

    print('Evaluating Test Data')
    print('Modulated')
    modulate_by_cla = True
    eval_by_cla(
        model,
        test_loader_cla,
        test_loader_cla_sl,
        device,
        modulate_by_cla,
        **kwargs)

    eval_by_loc(
        model,
        test_loader_loc,
        test_loader_loc_wl,
        device,
        modulate_by_cla,
        **kwargs)

    print('Unmodulated')
    modulate_by_cla = False
    eval_by_cla(
        model,
        test_loader_cla,
        test_loader_cla_sl,
        device,
        modulate_by_cla,
        **kwargs)

    eval_by_loc(
        model,
        test_loader_loc,
        test_loader_loc_wl,
        device,
        modulate_by_cla,
        **kwargs)
