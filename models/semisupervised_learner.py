import copy
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../optimizers')

from focal_loss import FocalLossBinary

class ChromatogramScaler(nn.Module):
    def __init__(
        self,
        num_channels=6,
        scale_independently=False,
        scale_precursors=False,
        lower=0.875,
        upper=1.125,
        device='cpu'):
        super(ChromatogramScaler, self).__init__()
        self.num_channels = num_channels
        self.scale_independently = scale_independently
        self.scale_precursors = scale_precursors
        self.lower = lower
        self.upper = upper
        self.device = device

    def forward(self, chromatogram_batch):
        if self.scale_independently:
            scaling_factors = (
                torch.FloatTensor(6, 1).uniform_(self.lower, self.upper))
        else:
            scaling_factors = (
                torch.FloatTensor(1).uniform_(self.lower, self.upper))

        chromatogram_batch[:, 0:6] = (
            chromatogram_batch[:, 0:6].to(self.device) * scaling_factors)

        if self.num_channels == 14:
            chromatogram_batch[:, 7:13] = (
                chromatogram_batch[:, 7:13].to(self.device) * scaling_factors)

        if self.scale_precursors:
            if self.num_channels == 14:
                scaling_factor = (
                    torch.FloatTensor(1).uniform_(self.lower, self.upper))
                chromatogram_batch[:, 13] = (
                    chromatogram_batch[:, 13].to(self.device) * scaling_factor)

        return chromatogram_batch

class ChromatogramShuffler(nn.Module):
    def __init__(self, num_channels=6):
        super(ChromatogramShuffler, self).__init__()
        self.num_channels = num_channels

    def forward(self, chromatogram_batch):
        shuffled_indices = torch.randperm(6)

        chromatogram_batch[:, 0:6] = (
            chromatogram_batch[:, 0:6][:, shuffled_indices])

        if self.num_channels == 14:
            chromatogram_batch[:, 7:13] = (
                chromatogram_batch[:, 7:13][:, shuffled_indices])

        return chromatogram_batch

class SemiSupervisedLearner(nn.Module):
    def __init__(
        self,
        model,
        wu=1,
        threshold=0.95,
        augmentator_num_channels=6,
        augmentator_scale_independently=False,
        augmentator_scale_precursors=False,
        augmentator_lower=0.875,
        augmentator_upper=1.125,
        augmentator_device='cpu',
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        debug=False):
        super(SemiSupervisedLearner, self).__init__()
        self.segmentator = copy.deepcopy(model)
        self.wu = wu
        self.threshold = threshold

        self.weak_augmentator = ChromatogramScaler(
            num_channels=augmentator_num_channels,
            scale_independently=augmentator_scale_independently,
            scale_precursors=augmentator_scale_precursors,
            lower=augmentator_lower,
            upper=augmentator_upper,
            device=augmentator_device
        )

        self.strong_augmentator = nn.Sequential(
            self.weak_augmentator,
            ChromatogramShuffler(num_channels=augmentator_num_channels)
        )

        self.loss = FocalLossBinary(
            loss_alpha, loss_gamma, loss_logits, loss_reduction
        )

        self.debug = debug

    def forward(self, unlabeled_batch, labeled_batch=None, labels=None):
        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'
            labeled_loss = torch.mean(
                self.loss(self.segmentator(labeled_batch), labels)
            )

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.segmentator(weakly_augmented)
            pseudo_labels = (weak_output >= self.threshold).float()
            quality_mask =  pseudo_labels.reshape(1, -1).squeeze()

            unlabeled_loss = torch.mean(
                quality_mask * 
                self.loss(self.segmentator(strongly_augmented), pseudo_labels)
            )
            
            if self.debug:
                print(
                    f'Labeled Loss: {labeled_loss.item()}, '
                    f'Unlabeled Loss: {unlabeled_loss.item()}, '
                    f'Weighted UL Loss: {self.wu * unlabeled_loss.item()}'
                )

            return labeled_loss + self.wu * unlabeled_loss
        else:
            return self.segmentator(unlabeled_batch)

class SemiSupervisedAlignmentLearner(SemiSupervisedLearner):
    def __init__(
        self,
        model,
        wu=1,
        threshold=0.85,
        augmentator_num_channels=6,
        augmentator_scale_independently=False,
        augmentator_scale_precursors=False,
        augmentator_lower=0.875,
        augmentator_upper=1.125,
        augmentator_device='cpu',
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        debug=False):
        super(SemiSupervisedAlignmentLearner, self).__init__(
            model,
            wu=wu,
            threshold=threshold,
            augmentator_num_channels=augmentator_num_channels,
            augmentator_scale_independently=augmentator_scale_independently,
            augmentator_scale_precursors=augmentator_scale_precursors,
            augmentator_lower=augmentator_lower,
            augmentator_upper=augmentator_upper,
            augmentator_device=augmentator_device,
            loss_alpha=loss_alpha,
            loss_gamma=loss_gamma,
            loss_logits=loss_logits,
            loss_reduction=loss_reduction,
            debug=debug
        )

    def forward(
        self,
        unlabeled_batch,
        template,
        template_label,
        labeled_batch=None,
        labels=None):
        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'
            labeled_loss = torch.mean(
                self.loss(
                    self.segmentator(labeled_batch, template, template_label),
                    labels
                )
            )

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.segmentator(weakly_augmented, template, template_label)
            pseudo_labels = (weak_output >= self.threshold).float()
            quality_mask =  pseudo_labels.reshape(1, -1).squeeze()

            unlabeled_loss = torch.mean(
                quality_mask * 
                self.loss(
                    self.segmentator(
                        strongly_augmented, template, template_label),
                    pseudo_labels
                )
            )
            
            if self.debug:
                print(
                    f'Labeled Loss: {labeled_loss.item()}, '
                    f'Unlabeled Loss: {unlabeled_loss.item()}, '
                    f'Weighted UL Loss: {self.wu * unlabeled_loss.item()}'
                )

            return labeled_loss + self.wu * unlabeled_loss
        else:
            return self.segmentator(unlabeled_batch, template, template_label)
