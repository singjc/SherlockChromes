import copy
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import erfinv

from models.modelzoo1d.dain import DAIN_Layer
from optimizers.focal_loss import FocalLossBinary

class ChromatogramScaler(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        scale_independently=False,
        scale_precursors=False,
        lower=0.875,
        upper=1.125,
        p=0.5,
        device='cpu'):
        super(ChromatogramScaler, self).__init__()
        self.mz_bins = mz_bins
        self.scale_independently = scale_independently
        self.scale_precursors = scale_precursors
        self.lower = lower
        self.upper = upper
        self.p = p
        self.device = device

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() < self.p:
            return chromatogram_batch

        if self.scale_independently:
            scaling_factors = (
                torch.FloatTensor(6, 1).uniform_(self.lower, self.upper)
            ).to(self.device)
            
            if self.mz_bins > 6:
                scaling_factors = (
                    scaling_factors.repeat_interleave(self.mz_bins // 6, dim=0)
                ).to(self.device)
        else:
            scaling_factors = (
                torch.FloatTensor(1).uniform_(self.lower, self.upper)
            ).to(self.device)

        chromatogram_batch[:, 0:self.mz_bins] = (
            chromatogram_batch[:, 0:self.mz_bins] * scaling_factors)

        if self.scale_precursors:
            scaling_factor = (
                torch.FloatTensor(1).uniform_(self.lower, self.upper)
            ).to(self.device)
            chromatogram_batch[:, self.mz_bins + 7:] = (
                chromatogram_batch[:, self.mz_bins + 7:] * scaling_factor)

        return chromatogram_batch

class ChromatogramJitterer(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        length=175,
        mean=0,
        std=1,
        p=0.5,
        device='cpu'):
        super(ChromatogramJitterer, self).__init__()
        self.mz_bins = mz_bins
        self.length = length
        self.mean = mean
        self.std = std
        self.p = p
        self.device = device

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() < self.p:
            return chromatogram_batch

        noise = (
            torch.FloatTensor(
                self.mz_bins, self.length
            ).normal_(self.mean, self.std)
        ).to(self.device)

        chromatogram_batch[:, 0:self.mz_bins]+= noise
        
        return chromatogram_batch

class ChromatogramShuffler(nn.Module):
    def __init__(self, mz_bins=6, p=0.5):
        super(ChromatogramShuffler, self).__init__()
        self.mz_bins = mz_bins
        self.p = p

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() < self.p:
            return chromatogram_batch

        shuffled_indices = torch.randperm(6)

        N = self.mz_bins // 6
        start, end = self.mz_bins + 1, self.mz_bins + 7

        if N == 1:
            chromatogram_batch[:, 0:self.mz_bins] = (
                chromatogram_batch[:, 0:self.mz_bins][:, shuffled_indices])
        else:
            b, M, n = chromatogram_batch.size()
            chromatogram_batch.reshape(
                b, M, -1, n)[:, shuffled_indices].reshape(b, -1, n)

        chromatogram_batch[:, start:end] = (
            chromatogram_batch[:, start:end][:, shuffled_indices])         

        return chromatogram_batch

class ChromatogramSpectraMasker(nn.Module):
    def __init__(self, mz_bins=6, F=1, m_F=1, p=0.5)
        super(ChromatogramSpectraMasker, self).__init__()
        self.v = mz_bins
        self.F = F
        self.m_F = m_F
        self.p = p

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() < self.p:
            return chromatogram_batch

        for i in range(self.m_F):
            f = torch.randint(0, self.F + 1, (1,)).item()
            f_0 = torch.randint(0, self.v - f, (1,)).item()
            chromatogram_batch[:, f_0:f] = 0

        return chromatogram_batch

class ChromatogramTimeMasker(nn.Module):
    def __init__(self, length=175, T=5, m_T=1, p=0.5)
        super(ChromatogramTimeMasker, self).__init__()
        self.tau = length
        self.T = T
        self.m_T = m_T
        self.p = p

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() < self.p:
            return chromatogram_batch

        for i in range(self.m_T):
            t = torch.randint(0, self.T + 1, (1,)).item()
            t_0 = torch.randint(0, self.tau - t, (1,)).item()
            chromatogram_batch[:, :, t_0:t] = 0

        return chromatogram_batch

class SemiSupervisedLearner1d(nn.Module):
    def __init__(
        self,
        model,
        wu=1,
        threshold=0.95,
        augmentator_p=0.5,
        augmentator_mz_bins=6,
        augmentator_scale_independently=False,
        augmentator_scale_precursors=False,
        augmentator_lower=0.875,
        augmentator_upper=1.125,
        augmentator_length=175,
        augmentator_mean=0,
        augmentator_std=1,
        augmentator_F=1,
        augmentator_m_F=1,
        augmentator_T=5,
        augmentator_m_T=1,
        normalize=False,
        normalization_mode='full',
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
        super(SemiSupervisedLearner1d, self).__init__()
        self.segmentator = copy.deepcopy(model)
        self.wu = wu
        self.threshold = threshold
        self.device = model_device

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=augmentator_mz_bins
            )
        else:
            self.normalization_layer = nn.Identity()

        self.weak_augmentator = ChromatogramScaler(
            mz_bins=augmentator_mz_bins,
            scale_independently=augmentator_scale_independently,
            scale_precursors=augmentator_scale_precursors,
            lower=augmentator_lower,
            upper=augmentator_upper,
            p=augmentator_p,
            device=self.device
        )

        self.strong_augmentator = nn.Sequential(
            self.weak_augmentator,
            ChromatogramJitterer(
                mz_bins=augmentator_mz_bins,
                length=augmentator_length,
                mean=augmentator_mean,
                std=augmentator_std,
                p=augmentator_p,
                device=self.device
            ),
            ChromatogramShuffler(mz_bins=augmentator_mz_bins, p=augmentator_p),
            ChromatogramSpectraMasker(
                mz_bins=augmentator_mz_bins,
                F=augmentator_F,
                m_F=augmentator_m_F,
                p=augmentator_p
            ),
            ChromatogramTimeMasker(
                length=augmentator_length,
                T=augmentator_T,
                m_T=augmentator_m_T,
                p=augmentator_p
            )
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

        if labeled_batch is not None:
            labeled_batch = self.normalization_layer(labeled_batch)

        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = unlabeled_batch[0:b_ul - 1]

            labeled_loss = torch.mean(
                self.loss(self.segmentator(labeled_batch), labels)
            )

            strongly_augmented = self.normalization_layer(
                self.strong_augmentator(unlabeled_batch)
            )
            weakly_augmented = self.normalization_layer(
                self.weak_augmentator(unlabeled_batch)
            )
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
            return self.segmentator(self.normalization_layer(unlabeled_batch))

class SemiSupervisedAlignmentLearner1d(SemiSupervisedLearner1d):
    def __init__(
        self,
        model,
        wu=1,
        threshold=0.85,
        augmentator_p=0.5,
        augmentator_mz_bins=6,
        augmentator_scale_independently=False,
        augmentator_scale_precursors=False,
        augmentator_lower=0.875,
        augmentator_upper=1.125,
        augmentator_length=175,
        augmentator_mean=0,
        augmentator_std=1,
        augmentator_F=1,
        augmentator_m_F=1,
        augmentator_T=5,
        augmentator_m_T=1,
        normalize=False,
        normalization_mode='full',
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
        super(SemiSupervisedAlignmentLearner1d, self).__init__(
            model,
            wu=wu,
            threshold=threshold,
            augmentator_p=augmentator_p,
            augmentator_mz_bins=augmentator_mz_bins,
            augmentator_scale_independently=augmentator_scale_independently,
            augmentator_scale_precursors=augmentator_scale_precursors,
            augmentator_lower=augmentator_lower,
            augmentator_upper=augmentator_upper,
            augmentator_length=augmentator_length,
            augmentator_mean=augmentator_mean,
            augmentator_std=augmentator_std,
            augmentator_F=augmentator_F,
            augmentator_m_F=augmentator_m_F,
            augmentator_T=augmentator_T,
            augmentator_m_T=augmentator_m_T,
            normalize=normalize,
            normalization_mode=normalization_mode,
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
        templates,
        template_labels,
        labeled_batch=None,
        labels=None):
        b_ul, c_ul, l_ul = unlabeled_batch.size()

        templates = self.normalization_layer(templates)

        if labeled_batch is not None:
            labeled_batch = self.normalization_layer(labeled_batch)

        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = unlabeled_batch[0:b_ul - 1]
                
            labeled_loss = torch.mean(
                self.loss(
                    self.segmentator(labeled_batch, templates, template_labels),
                    labels
                )
            )

            strongly_augmented = self.normalization_layer(
                self.strong_augmentator(unlabeled_batch)
            )
            weakly_augmented = self.normalization_layer(
                self.weak_augmentator(unlabeled_batch)
            )
            weak_output = self.to_out(
                self.segmentator(weakly_augmented, templates, template_labels)
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
                        strongly_augmented, templates, template_labels),
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
            return self.segmentator(
                self.normalization_layer(unlabeled_batch),
                templates,
                template_labels
            )
