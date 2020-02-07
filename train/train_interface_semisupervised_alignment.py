import sys
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')
sys.path.insert(0, '../optimizers')

from semisupervised_learner import SemiSupervisedAlignmentLearner
from train_semisupervised_alignment import train

def main(
        data,
        template_data,
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
        'weak_p',
        'weak_drop_channels',
        'strong_p',
        'strong_drop_channels',
        'loss_alpha',
        'loss_gamma',
        'loss_logits',
        'loss_reduction',
        'debug']:
        if kw in train_kwargs:
            model_kwargs[kw] = train_kwargs[kw]

    model = SemiSupervisedAlignmentLearner(
        model,
        **model_kwargs)

    optimizer = optim.AdamW(model.parameters(), **optimizer_kwargs)

    train(
        data,
        template_data,
        model,
        optimizer,
        loss,
        sampling_fn,
        collate_fn,
        device,
        **train_kwargs)
