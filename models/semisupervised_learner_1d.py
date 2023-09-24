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
        augment_precursor=False,
        scale_independently=False,
        lower=0.875,
        upper=1.125,
        p=0.5,
        device='cpu'
    ):
        super(ChromatogramScaler, self).__init__()
        self.mz_bins = mz_bins
        self.num_factors = 6
        self.scale_independently = scale_independently
        self.lower = lower
        self.upper = upper
        self.p = p
        self.device = device

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6
            self.num_factors += 1

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        if self.scale_independently:
            scaling_factors = (
                torch.FloatTensor(
                    self.num_factors, 1).uniform_(self.lower, self.upper)
            ).to(self.device)

            if self.mz_bins > 6:
                scaling_factors = (
                    scaling_factors.repeat_interleave(
                        self.mz_bins // self.num_factors, dim=0)
                ).to(self.device)
        else:
            scaling_factors = (
                torch.FloatTensor(1).uniform_(self.lower, self.upper)
            ).to(self.device)

        chromatogram_batch[:, 0:self.mz_bins] = (
            chromatogram_batch[:, 0:self.mz_bins] * scaling_factors)

        return chromatogram_batch


class ChromatogramJitterer(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=False,
        length=175,
        mean=0,
        std=1,
        p=0.5,
        device='cpu'
    ):
        super(ChromatogramJitterer, self).__init__()
        self.mz_bins = mz_bins
        self.length = length
        self.mean = mean
        self.std = std
        self.p = p
        self.device = device

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        noise = (
            torch.FloatTensor(
                self.mz_bins, self.length
            ).normal_(self.mean, self.std)
        ).to(self.device)

        chromatogram_batch[:, 0:self.mz_bins] = (
            chromatogram_batch[:, 0:self.mz_bins] + noise)

        return chromatogram_batch


class ChromatogramShuffler(nn.Module):
    def __init__(self, mz_bins=6, p=0.5):
        super(ChromatogramShuffler, self).__init__()
        self.mz_bins = mz_bins
        self.p = p

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        shuffled_indices = torch.randperm(6)

        M = self.mz_bins // 6
        start = self.mz_bins + M
        end = start + 6

        if M == 1:
            chromatogram_batch[:, 0:self.mz_bins] = (
                chromatogram_batch[:, 0:self.mz_bins][:, shuffled_indices])
        else:
            b, _, n = chromatogram_batch.size()
            chromatogram_batch[:, 0:self.mz_bins] = (
                chromatogram_batch[:, 0:self.mz_bins].reshape(
                    b, 6, M, n)[:, shuffled_indices].reshape(b, -1, n)
            )

        chromatogram_batch[:, start:end] = (
            chromatogram_batch[:, start:end][:, shuffled_indices])

        return chromatogram_batch


class ChromatogramSpectraMasker(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=False,
        F=1,
        m_F=1,
        p=0.5
    ):
        super(ChromatogramSpectraMasker, self).__init__()
        self.v = mz_bins
        self.F = F
        self.m_F = m_F
        self.p = p

        if augment_precursor:
            self.v += self.v // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        for i in range(self.m_F):
            f = torch.randint(0, self.F + 1, (1,)).item()
            f_0 = torch.randint(0, self.v - f, (1,)).item()
            chromatogram_batch[:, f_0:f] = 0

        return chromatogram_batch


class ChromatogramTimeMasker(nn.Module):
    def __init__(
        self,
        mz_bins=6,
        augment_precursor=False,
        length=175,
        T=5,
        m_T=1,
        p=0.5
    ):
        super(ChromatogramTimeMasker, self).__init__()
        self.mz_bins = mz_bins
        self.tau = length
        self.T = T
        self.m_T = m_T
        self.p = p

        if augment_precursor:
            self.mz_bins += self.mz_bins // 6

    def forward(self, chromatogram_batch):
        if torch.rand(1).item() > self.p:
            return chromatogram_batch

        for i in range(self.m_T):
            t = torch.randint(0, self.T + 1, (1,)).item()
            t_0 = torch.randint(0, self.tau - t, (1,)).item()
            chromatogram_batch[:, 0:self.mz_bins, t_0:t] = 0

        return chromatogram_batch


class ChromatogramNormalizer(nn.Module):
    def __init__(self, mz_bins=6, standardize=False):
        super(ChromatogramNormalizer, self).__init__()
        self.mz_bins = mz_bins
        self.standardize = standardize

    def forward(self, chromatogram_batch):
        M = self.mz_bins // 6
        start = self.mz_bins + M
        end = start + 6

        if self.standardize:
            sigma, mu = torch.std_mean(
                torch.cat(
                    [chromatogram_batch[:, 0:self.mz_bins],
                    chromatogram_batch[:, start:end]], dim=1
                ),
                dim=1,
                keepdim=True
            )
            sigma += 1e-7
            chromatogram_batch[:, 0:self.mz_bins] = (
                (chromatogram_batch[:, 0:self.mz_bins] - mu) / sigma)
            chromatogram_batch[:, start:end] = (
                (chromatogram_batch[:, start:end] - mu) / sigma)

            sigma, mu = torch.std_mean(
                chromatogram_batch[:, self.mz_bins:start], dim=1, keepdim=True)
            sigma += 1e-7
            chromatogram_batch[:, self.mz_bins:start] = (
                (chromatogram_batch[:, self.mz_bins:start] - mu) / sigma)
        else:
            x = torch.cat(
                [chromatogram_batch[:, 0:self.mz_bins],
                chromatogram_batch[:, start:end]],
                dim=1
            )
            x_min, _ = torch.min(x, dim=1, keepdim=True)
            x_max, _ = torch.max(x, dim=1, keepdim=True)
            x_max += 1e-7
            chromatogram_batch[:, 0:self.mz_bins] = (
                (chromatogram_batch[:, 0:self.mz_bins] - x_min) /
                (x_max - x_min))
            chromatogram_batch[:, start:end] = (
                (chromatogram_batch[:, start:end] - x_min) /
                (x_max - x_min))

            x_min, _ = torch.min(
                chromatogram_batch[:, self.mz_bins:start], dim=1, keepdim=True)
            x_max, _ = torch.max(
                chromatogram_batch[:, self.mz_bins:start], dim=1, keepdim=True)
            x_max += 1e-7
            chromatogram_batch[:, self.mz_bins:start] = (
                (chromatogram_batch[:, self.mz_bins:start] - x_min) / 
                (x_max - x_min))

        x_min = torch.min(chromatogram_batch[:, -2:-1])
        x_max = torch.max(chromatogram_batch[:, -2:-1]) + 1e-7
        chromatogram_batch[:, -2:-1] = (
                (chromatogram_batch[:, -2:-1] - x_min) / (x_max - x_min))

        x_min, x_max = 1, 3
        chromatogram_batch[:, -1:] = (
                (chromatogram_batch[:, -1:] - x_min) / 
                (x_max - x_min))

        return chromatogram_batch


class SemiSupervisedLearner1d(nn.Module):
    def __init__(
        self,
        model,
        semisupervised=True,
        wu=1,
        threshold=0.95,
        use_weak_labels=False,
        enforce_weak_consistency=False,
        enforce_sparse_loc=False,
        enforce_sparse_attn=False,
        sparsity_modulator=1e-4,
        augmentator_p=0.5,
        augmentator_mz_bins=6,
        augmentator_augment_precursor=False,
        augmentator_scale_independently=False,
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
        normalization_channels=6,
        regularizer_mode='none',
        regularizer_sigma_min=4,
        regularizer_sigma_max=16,
        regularizer_p_min=0.5,
        regularizer_p_max=0.5,
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        model_device='cpu',
        debug=False,
        save_normalized=False
    ):
        super(SemiSupervisedLearner1d, self).__init__()
        self.model = copy.deepcopy(model)
        self.semisupervised = semisupervised
        self.wu = wu
        self.threshold = threshold
        self.use_weak_labels = use_weak_labels
        self.enforce_weak_consistency = enforce_weak_consistency
        self.enforce_sparse_loc = enforce_sparse_loc
        self.enforce_sparse_attn = enforce_sparse_attn
        self.sparsity_modulator = sparsity_modulator
        self.device = model_device

        self.weak_augmentator = ChromatogramScaler(
            mz_bins=augmentator_mz_bins,
            augment_precursor=augmentator_augment_precursor,
            scale_independently=False,
            lower=augmentator_lower,
            upper=augmentator_upper,
            p=1,
            device=self.device
        )

        self.strong_augmentator = nn.Sequential(
            ChromatogramScaler(
                mz_bins=augmentator_mz_bins,
                augment_precursor=augmentator_augment_precursor,
                scale_independently=augmentator_scale_independently,
                lower=augmentator_lower,
                upper=augmentator_upper,
                p=augmentator_p,
                device=self.device
            ),
            ChromatogramJitterer(
                mz_bins=augmentator_mz_bins,
                augment_precursor=augmentator_augment_precursor,
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
                mz_bins=augmentator_mz_bins,
                augment_precursor=augmentator_augment_precursor,
                length=augmentator_length,
                T=augmentator_T,
                m_T=augmentator_m_T,
                p=augmentator_p
            )
        )

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=normalization_channels
            )
        else:
            self.normalization_layer = nn.Identity()

        self.regularizer_mode = regularizer_mode
        self.regularizer_sigma_min = regularizer_sigma_min
        self.regularizer_sigma_max = regularizer_sigma_max
        self.regularizer_p_min = regularizer_p_min
        self.regularizer_p_max = regularizer_p_max

        self.loss = FocalLossBinary(
            alpha=loss_alpha,
            gamma=loss_gamma,
            logits=loss_logits,
            reduction=loss_reduction
        )

        self.debug = debug
        self.save_normalized = save_normalized
        self.normalized = None

    def get_normalized(self):
        return self.normalized

    def get_model(self):
        model = self.model

        if (
            'normalization_layer' in [n for n, m in model.named_modules()] and
            isinstance(model.normalization_layer, nn.Identity)
        ):
            model.normalization_layer = self.normalization_layer

        return model

    def generate_zebra_mask(
        self,
        length,
        sigma_min=4,
        sigma_max=16,
        p_min=0.5,
        p_max=0.5
    ):
        sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))
        p = np.random.uniform(p_min, p_max)
        noise_image = np.random.normal(size=length)
        noise_image_smoothed = gaussian_filter1d(noise_image, sigma)
        threshold = (
            erfinv((p * 2) - 1) * (2 ** 0.5) * noise_image_smoothed.std()
            + noise_image_smoothed.mean())

        return (noise_image_smoothed > threshold).astype(float)

    def forward(self, unlabeled_batch, labeled_batch=None, labels=None):
        if self.training:
            assert labeled_batch is not None, 'missing labeled data!'
            assert labels is not None, 'missing labels!'

            b_ul, c_ul, l_ul = unlabeled_batch.size()
            labeled_batch = self.normalization_layer(labeled_batch)

            if self.regularizer_mode == 'cutmix':
                if b_ul % 2 != 0:
                    unlabeled_batch = torch.cat(
                        [
                            unlabeled_batch,
                            torch.zeros(1, c_ul, l_ul).to(self.device)
                        ],
                        dim=0
                    )
                    b_ul += 1

            orig_setting = self.model.output_mode

            self.model.output_mode = 'all'
            out_dict = self.model(labeled_batch)
            weak_output, strong_output, attn = (
                out_dict['cla'],
                out_dict['loc'],
                out_dict['attn']
            )

            if self.use_weak_labels:
                labeled_loss = torch.mean(
                    self.loss(weak_output, labels)
                )
            else:
                weak_labeled_loss = 0

                if self.enforce_weak_consistency:
                    weak_labeled_loss = torch.mean(
                        self.loss(
                            weak_output,
                            torch.max(labels, dim=1, keepdim=True)[0]
                        )
                    )

                labeled_loss = (
                    torch.mean(self.loss(strong_output, labels))
                    + weak_labeled_loss
                )

            if self.enforce_sparse_loc:
                labeled_loss = labeled_loss + torch.mean(
                    torch.norm(strong_output, p=1, dim=1)
                    * self.sparsity_modulator)

            if self.enforce_sparse_attn:
                labeled_loss = labeled_loss + torch.mean(
                    torch.norm(attn, p=1, dim=1) * self.sparsity_modulator)

            if self.semisupervised:
                strongly_augmented = self.normalization_layer(
                    self.strong_augmentator(unlabeled_batch)
                )
                weakly_augmented = self.normalization_layer(
                    self.weak_augmentator(unlabeled_batch)
                )

                if self.enforce_weak_consistency:
                    self.model.output_mode = 'all'
                    out_dict = self.model(weakly_augmented)
                    weak_output, strong_output = (
                        out_dict['cla'],
                        out_dict['loc']
                    )

                    weak_pseudo_labels = (weak_output >= 0.5).float()
                    weak_quality_modulator = (
                        (weak_output >= self.threshold).float()
                        + (weak_output <= (1 - self.threshold))
                    ).reshape(1, -1).squeeze()

                    # Variable required for cutmix
                    lam = None
                else:
                    self.model.output_mode = 'loc'
                    strong_output = self.model(strongly_augmented)

                strong_pseudo_labels = (strong_output >= 0.5).float()
                strong_quality_modulator = (
                    (strong_output >= self.threshold).float()
                    + (strong_output <= (1 - self.threshold))
                )

                # Variable required for cutmix
                b_ul_half = b_ul // 2

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
                        strongly_augmented = (
                            strongly_augmented * regularizer_mask)
                        strong_quality_modulator = (
                            strong_quality_modulator * regularizer_mask)
                    elif self.regularizer_mode == 'cutmix':
                        if self.enforce_weak_consistency:
                            lam = (
                                torch.sum(regularizer_mask)
                                / regularizer_mask.nelement()
                            )

                        strongly_augmented = (
                            (
                                strongly_augmented[0:b_ul_half]
                                * regularizer_mask)
                            + (strongly_augmented[b_ul_half:]
                                * (1 - regularizer_mask)))
                        strong_pseudo_labels = (
                            (
                                strong_pseudo_labels[0:b_ul_half]
                                * regularizer_mask)
                            + (strong_pseudo_labels[b_ul_half:]
                                * (1 - regularizer_mask)))
                        strong_quality_modulator = (
                            (strong_quality_modulator[0:b_ul_half]
                                * regularizer_mask)
                            + (strong_quality_modulator[b_ul_half:]
                                * (1 - regularizer_mask)))

                strong_quality_modulator = torch.mean(
                    strong_quality_modulator.reshape(1, -1).squeeze())

                self.model.output_mode = 'all'
                out_dict = self.model(strongly_augmented)
                weak_output, strong_output, attn = (
                    out_dict['cla'],
                    out_dict['loc'],
                    out_dict['attn']
                )

                if (
                    self.enforce_weak_consistency
                    and self.regularizer_mode == 'cutmix'
                ):
                    weak_unlabeled_loss_a = lam * torch.mean(
                        self.loss(
                            weak_output,
                            weak_pseudo_labels[:b_ul_half]
                        )[weak_quality_modulator[:b_ul_half].bool()]
                    )

                    if torch.isnan(weak_unlabeled_loss_a):
                        weak_unlabeled_loss_a = 0.0

                    weak_unlabeled_loss_b = (1 - lam) * torch.mean(
                        self.loss(
                            weak_output,
                            weak_pseudo_labels[b_ul_half:]
                        )[weak_quality_modulator[b_ul_half:].bool()]
                    )

                    if torch.isnan(weak_unlabeled_loss_b):
                        weak_unlabeled_loss_b = 0.0

                    weak_unlabeled_loss = (
                        weak_unlabeled_loss_a + weak_unlabeled_loss_b
                    )
                elif self.enforce_weak_consistency:
                    weak_unlabeled_loss = torch.mean(
                        self.loss(
                            weak_output,
                            weak_pseudo_labels
                        )[weak_quality_modulator.bool()]
                    )

                    if torch.isnan(weak_unlabeled_loss):
                        weak_unlabeled_loss = 0.0

                strong_unlabeled_loss = strong_quality_modulator * torch.mean(
                    self.loss(strong_output, strong_pseudo_labels)
                )

                self.model.output_mode = orig_setting

                if self.enforce_weak_consistency:
                    unlabeled_loss = (
                        weak_unlabeled_loss + strong_unlabeled_loss)
                else:
                    unlabeled_loss = strong_unlabeled_loss

                if self.enforce_sparse_loc:
                    unlabeled_loss = (
                        unlabeled_loss
                        + strong_quality_modulator * torch.mean(
                            torch.norm(strong_output, p=1, dim=1)
                            * self.sparsity_modulator))

                if self.enforce_sparse_attn:
                    unlabeled_loss = (
                        unlabeled_loss
                        + strong_quality_modulator * torch.mean(
                            torch.norm(attn, p=1, dim=1)
                            * self.sparsity_modulator))
            else:
                unlabeled_loss = 0

            if self.debug:
                if self.semisupervised:
                    if self.use_weak_labels:
                        num_positive = int(torch.sum(labels).item())
                    else:
                        num_positive = 'n/a'

                    if self.enforce_weak_consistency:
                        if isinstance(weak_unlabeled_loss, float):
                            weak_unlabeled_loss_debug = weak_unlabeled_loss
                        else:
                            weak_unlabeled_loss_debug = (
                                weak_unlabeled_loss.item())

                        weak_unlabeled_loss_debug = (
                            f'{weak_unlabeled_loss_debug:.8f}')
                        weak_quality_modulator_debug = torch.mean(
                            weak_quality_modulator).item()
                        weak_quality_modulator_debug = (
                            f'{weak_quality_modulator_debug:.8f}')
                    else:
                        weak_unlabeled_loss_debug = 'n/a'
                        weak_quality_modulator_debug = 'n/a'

                    print(
                        f'L Loss: {labeled_loss.item():.8f}, '
                        f'# Positive: {num_positive}, '
                        f'UL Loss: {unlabeled_loss.item():.8f}, '
                        'Weak Quality Modulator u: '
                        f'{weak_quality_modulator_debug}, '
                        f'Weak UL Loss: {weak_unlabeled_loss_debug}, '
                        'Strong Quality Modulator u: '
                        f'{strong_quality_modulator.item():.8f}, '
                        f'Strong UL Loss: {strong_unlabeled_loss.item():.8f}, '
                        'Weighted UL Loss: '
                        f'{self.wu * unlabeled_loss.item():.8f}'
                    )
                else:
                    print(f'L Loss: {labeled_loss.item():.8f}')

            return labeled_loss + self.wu * unlabeled_loss
        else:
            normalized = self.normalization_layer(unlabeled_batch)

            if self.save_normalized:
                self.normalized = normalized

            return self.model(normalized)


# TODO: Update forward to match parent structure
class SemiSupervisedAlignmentLearner1d(SemiSupervisedLearner1d):
    def __init__(
        self,
        model,
        semisupervised=True,
        wu=1,
        threshold=0.85,
        use_weak_labels=False,
        enforce_weak_consistency=False,
        enforce_sparse_loc=False,
        enforce_sparse_attn=False,
        sparsity_modulator=1e-4,
        augmentator_p=0.5,
        augmentator_mz_bins=6,
        augmentator_augment_precursor=False,
        augmentator_scale_independently=False,
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
        normalization_channels=6,
        regularizer_mode='none',
        regularizer_sigma_min=4,
        regularizer_sigma_max=16,
        regularizer_p_min=0.5,
        regularizer_p_max=0.5,
        loss_alpha=0.25,
        loss_gamma=2,
        loss_logits=False,
        loss_reduction='none',
        model_device='cpu',
        debug=False,
        save_normalized=False
    ):
        super(SemiSupervisedAlignmentLearner1d, self).__init__(
            model,
            semisupervised=semisupervised,
            wu=wu,
            threshold=threshold,
            use_weak_labels=use_weak_labels,
            enforce_weak_consistency=enforce_weak_consistency,
            enforce_sparse_loc=enforce_sparse_loc,
            enforce_sparse_attn=enforce_sparse_attn,
            sparsity_modulator=sparsity_modulator,
            augmentator_p=augmentator_p,
            augmentator_mz_bins=augmentator_mz_bins,
            augmentator_augment_precursor=augmentator_augment_precursor,
            augmentator_scale_independently=augmentator_scale_independently,
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
            normalization_channels=normalization_channels,
            regularizer_mode=regularizer_mode,
            regularizer_sigma_min=regularizer_sigma_min,
            regularizer_sigma_max=regularizer_sigma_max,
            regularizer_p_min=regularizer_p_min,
            regularizer_p_max=regularizer_p_max,
            loss_alpha=loss_alpha,
            loss_gamma=loss_gamma,
            loss_logits=loss_logits,
            loss_reduction=loss_reduction,
            model_device=model_device,
            debug=debug,
            save_normalized=save_normalized
        )

    def forward(
        self,
        unlabeled_batch,
        templates,
        template_labels,
        labeled_batch=None,
        labels=None
    ):
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

            self.model.aggregate_output = self.use_weak_labels

            labeled_loss = torch.mean(
                self.loss(
                    self.model(labeled_batch, templates, template_labels),
                    labels
                )
            )

            self.model.aggregate_output = False

            strongly_augmented = self.normalization_layer(
                self.strong_augmentator(unlabeled_batch)
            )
            weakly_augmented = self.normalization_layer(
                self.weak_augmentator(unlabeled_batch)
            )
            weak_output = self.to_out(
                self.model(weakly_augmented, templates, template_labels)
            )
            pseudo_labels = (weak_output >= 0.5).float()
            quality_modulator = (
                (weak_output >= self.threshold).float()
                + (weak_output <= (1 - self.threshold))
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

            quality_modulator = torch.mean(quality_modulator)

            unlabeled_loss = torch.mean(
                quality_modulator *
                self.loss(
                    self.model(
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
            normalized = self.normalization_layer(unlabeled_batch)

            if self.save_normalized:
                self.normalized = normalized

            return self.to_out(
                self.model(normalized, templates, template_labels))
