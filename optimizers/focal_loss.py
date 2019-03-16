import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossBinary(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduction='mean'):
        super(FocalLossBinary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(
                inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss


if __name__ == "__main__":
    pass
