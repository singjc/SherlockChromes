import torch
import torch.nn as nn


class DummyLoss(nn.Module):
    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self, inputs, targets):
        return inputs
