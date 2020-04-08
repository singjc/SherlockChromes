import copy
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import erfinv

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
                torch.FloatTensor(6, 1).uniform_(self.lower, self.upper)
            ).to(self.device)
        else:
            scaling_factors = (
                torch.FloatTensor(1).uniform_(self.lower, self.upper)
            ).to(self.device)

        chromatogram_batch[:, 0:6] = (
            chromatogram_batch[:, 0:6].to(self.device) * scaling_factors)

        if self.num_channels == 14:
            chromatogram_batch[:, 7:13] = (
                chromatogram_batch[:, 7:13].to(self.device) * scaling_factors)

        if self.scale_precursors:
            if self.num_channels == 14:
                scaling_factor = (
                    torch.FloatTensor(1).uniform_(self.lower, self.upper)
                ).to(self.device)
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
        regularizer_mode='none',
        regularizer_sigma_min=4,
        regularizer_sigma_max=16,
        regularizer_p_min=0.5,
        regularizer_p_max=0.5,
        modulation_mode='mask',
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        model_device='cpu',
        debug=False):
        super(SemiSupervisedLearner, self).__init__()
        self.segmentator = copy.deepcopy(model)
        self.wu = wu
        self.threshold = threshold
        
        self.device = model_device

        self.weak_augmentator = ChromatogramScaler(
            num_channels=augmentator_num_channels,
            scale_independently=augmentator_scale_independently,
            scale_precursors=augmentator_scale_precursors,
            lower=augmentator_lower,
            upper=augmentator_upper,
            device=self.device
        )

        self.strong_augmentator = nn.Sequential(
            self.weak_augmentator,
            ChromatogramShuffler(num_channels=augmentator_num_channels)
        )

        self.regularizer_mode = regularizer_mode
        self.regularizer_sigma_min = regularizer_sigma_min
        self.regularizer_sigma_max = regularizer_sigma_max
        self.regularizer_p_min = regularizer_p_min
        self.regularizer_p_max = regularizer_p_max

        self.modulation_mode = modulation_mode

        self.loss = FocalLossBinary(
            alpha=loss_alpha,
            gamma=loss_gamma,
            logits=loss_logits,
            reduction=loss_reduction
        )

        if loss_logits:
            self.to_out = nn.Sigmoid()
        else:
            self.to_out = nn.Identity()

        self.debug = debug

    def generate_zebra_mask(
        self,
        length,
        sigma_min=4,
        sigma_max=16,
        p_min=0.5,
        p_max=0.5):
        sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))
        p = np.random.uniform(p_min, p_max)
        noise_image = np.random.normal(size=length)
        noise_image_smoothed = gaussian_filter1d(noise_image, sigma)
        threshold = (
            erfinv(p*2 - 1) * (2**0.5) * noise_image_smoothed.std() + 
            noise_image_smoothed.mean()
        )

        return (noise_image_smoothed > threshold).astype(float)

    def forward(self, unlabeled_batch, labeled_batch=None, labels=None):
        b_ul, c_ul, l_ul = unlabeled_batch.size()

        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = unlabeled_batch[0:b_ul - 1]

            labeled_loss = torch.mean(
                self.loss(self.segmentator(labeled_batch), labels)
            )

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.to_out(self.segmentator(weakly_augmented))
            pseudo_labels = (weak_output >= 0.5).float()
            quality_modulator =  (
                (weak_output >= self.threshold).float() + 
                (weak_output <= (1 - self.threshold))
            )

            if self.regularizer_mode != 'none':
                regularizer_mask = torch.from_numpy(
                    self.generate_zebra_mask(
                        l_ul,
                        sigma_min=self.regularizer_sigma_min,
                        sigma_max=self.regularizer_sigma_max,
                        p_min=self.regularizer_p_min,
                        p_max=self.regularizer_p_max
                    )
                ).float().to(self.device)

                if self.regularizer_mode == 'cutout':
                    strongly_augmented = strongly_augmented * regularizer_mask
                    quality_modulator = quality_modulator * regularizer_mask
                elif self.regularizer_mode == 'cutmix':
                    b_ul_half = b_ul // 2
                    strongly_augmented = (
                        (
                            strongly_augmented[0:b_ul_half].to(self.device) * 
                            regularizer_mask
                        ) +
                        (
                            strongly_augmented[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )
                    pseudo_labels = (
                        (
                            pseudo_labels[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            pseudo_labels[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )
                    quality_modulator = (
                        (
                            quality_modulator[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            quality_modulator[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )

            quality_modulator = quality_modulator.reshape(1, -1).squeeze()

            if self.modulation_mode == 'mean':
                quality_modulator = torch.mean(quality_modulator)

            unlabeled_loss = torch.mean(
                quality_modulator * 
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
        regularizer_mode='none',
        regularizer_sigma_min=4,
        regularizer_sigma_max=16,
        regularizer_p_min=0.5,
        regularizer_p_max=0.5,
        modulation_mode='mask',
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        model_device='cpu',
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
            regularizer_mode=regularizer_mode,
            regularizer_sigma_min=regularizer_sigma_min,
            regularizer_sigma_max=regularizer_sigma_max,
            regularizer_p_min=regularizer_p_min,
            regularizer_p_max=regularizer_p_max,
            modulation_mode=modulation_mode,
            loss_alpha=loss_alpha,
            loss_gamma=loss_gamma,
            loss_logits=loss_logits,
            loss_reduction=loss_reduction,
            model_device=model_device,
            debug=debug
        )

    def forward(
        self,
        unlabeled_batch,
        template,
        template_label,
        labeled_batch=None,
        labels=None):
        b_ul, c_ul, l_ul = unlabeled_batch.size()

        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = unlabeled_batch[0:b_ul - 1]
                
            labeled_loss = torch.mean(
                self.loss(
                    self.segmentator(labeled_batch, template, template_label),
                    labels
                )
            )

            strongly_augmented = self.strong_augmentator(unlabeled_batch)
            weakly_augmented = self.weak_augmentator(unlabeled_batch)
            weak_output = self.to_out(
                self.segmentator(weakly_augmented, template, template_label)
            )
            pseudo_labels = (weak_output >= 0.5).float()
            quality_modulator =  (
                (weak_output >= self.threshold).float() + 
                (weak_output <= (1 - self.threshold))
            )

            if self.regularizer_mode != 'none':
                regularizer_mask = torch.from_numpy(
                    self.generate_zebra_mask(
                        l_ul,
                        sigma_min=self.regularizer_sigma_min,
                        sigma_max=self.regularizer_sigma_max,
                        p_min=self.regularizer_p_min,
                        p_max=self.regularizer_p_max
                    )
                ).float().to(self.device)

                if self.regularizer_mode == 'cutout':
                    strongly_augmented = strongly_augmented * regularizer_mask
                    quality_modulator = quality_modulator * regularizer_mask
                elif self.regularizer_mode == 'cutmix':
                    b_ul_half = b_ul // 2
                    strongly_augmented = (
                        (
                            strongly_augmented[0:b_ul_half].to(self.device) * 
                            regularizer_mask
                        ) +
                        (
                            strongly_augmented[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )
                    pseudo_labels = (
                        (
                            pseudo_labels[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            pseudo_labels[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )
                    quality_modulator = (
                        (
                            quality_modulator[0:b_ul_half].to(self.device) *
                            regularizer_mask
                        ) +
                        (
                            quality_modulator[b_ul_half:].to(self.device) *
                            (1 - regularizer_mask)
                        )
                    )

            quality_modulator = quality_modulator.reshape(1, -1).squeeze()

            if self.modulation_mode == 'mean':
                quality_modulator = torch.mean(quality_modulator)

            unlabeled_loss = torch.mean(
                quality_modulator * 
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
