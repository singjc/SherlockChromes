# Adapted from https://github.com/gpleiss/temperature_scaling
import torch
import torch.nn as nn

class ECELossBinary(nn.Module):
    """
    Calculates the Expected Calibration Error of a binary model.
    (This isn't necessary for temperature scaling, just a cool metric).

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15, logits=False):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELossBinary, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.logits = logits
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            inputs = inputs.reshape(1, -1)
            inputs = inputs.squeeze()
            targets = targets.reshape(1, -1)
            targets = targets.squeeze()
            
        if self.logits:
            inputs = self.sigmoid(inputs)

        confidences = inputs
        predictions = (inputs >= 0.5).float()
        accuracies = predictions.eq(targets)

        ece = torch.zeros(1, device=inputs.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece