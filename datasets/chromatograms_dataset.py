import numpy as np
import torch

from torch.utils.data import Dataset

class ChromatogramsInMemoryDataset(Dataset):
    """Chromatograms stored in memory dataset."""

    def __init__(self, chromatograms, labels, transform=None):
        """
        Args:
            chromatograms (string): Path to the npy file with chromatograms.
            labels (string): Path to the npy file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.chromatograms = np.load(chromatograms)
        self.labels = np.load(labels)
        self.transform = transform
    
    def __len__(self):
        return chromatograms.shape[0]

    def __getitem__(self, idx):
        chromatogram = torch.from_numpy(self.chromatograms[idx]).float()
        labels = torch.from_numpy(self.labels[idx]).float()
        sample = {'chromatogram': chromatogram, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
