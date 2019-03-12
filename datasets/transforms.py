import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        chromatogram = torch.from_numpy(sample[0]).float()
        label = torch.from_numpy(sample[1]).float()

        return chromatogram, label
        