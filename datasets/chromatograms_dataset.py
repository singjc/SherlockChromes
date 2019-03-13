import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset

class ChromatogramsDataset(Dataset):
    """Chromatograms stored in memory dataset."""

    def __init__(self, root_path, chromatograms, labels, transform=None):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram filenames.
            labels (string): Filename of the npy file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.labels = np.load(os.path.join(self.root_dir, labels))
        self.transform = transform
    
    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 0]) + '.npy'
        chromatogram = np.load(chromatogram_name).astype(float)
        label = self.labels[idx].astype(float)

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label
