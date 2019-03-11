import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset

class ChromatogramsDataset(Dataset):
    """Chromatograms stored in memory dataset."""

    def __init__(self, root_dir, chromatograms, labels, transform=None):
        """
        Args:
            root_dir (string): Path to the folder of chromatogram npy files.
            chromatograms (string): Filename of CSV with chromatogram filenames.
            labels (string): Filename of the npy file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_dir
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         self.chromatograms))
        self.labels = np.load(os.path.join(self.root_dir, labels))
        self.transform = transform
    
    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 0])
        chromatogram = torch.from_numpy(np.load(chromatogram_name)).float()
        labels = torch.from_numpy(self.labels[idx]).float()
        sample = {'chromatogram': chromatogram, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
