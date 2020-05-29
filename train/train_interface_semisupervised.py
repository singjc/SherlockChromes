import sys
import torch.optim as optim

from models.semisupervised_learner_1d import SemiSupervisedLearner1d
from train.train_semisupervised import train

def main(
        data,
        model,
        loss,
        sampling_fn,
        collate_fn,
        optimizer_kwargs,
        train_kwargs,
        device):
    model_kwargs = {}
    for kw in [
        'wu',
        'threshold',
        'augmentator_p',
        'augmentator_mz_bins',
        'augmentator_augment_precursor',
        'augmentator_scale_independently',
        'augmentator_lower',
        'augmentator_upper',
        'augmentator_length',
        'augmentator_mean',
        'augmentator_std',
        'augmentator_F',
        'augmentator_m_F',
        'augmentator_T',
        'augmentator_m_T',
        'normalize',
        'normalization_mode',
        'normalization_channels',
        'regularizer_mode',
        'regularizer_sigma_min',
        'regularizer_sigma_max',
        'regularizer_p_min',
        'regularizer_p_max',
        'modulation_mode',
        'loss_alpha',
        'loss_gamma',
        'loss_logits',
        'loss_reduction',
        'model_device',
        'debug']:
        if kw in train_kwargs:
            model_kwargs[kw] = train_kwargs[kw]

    model = SemiSupervisedLearner1d(
        model,
        **model_kwargs)

    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    train(
        data,
        model,
        optimizer,
        loss,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)
