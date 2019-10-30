import numpy as np
import os
import pandas as pd
import torch

from torch.utils.data import Dataset

class ChromatogramsDataset(Dataset):
    """Whole Chromatograms dataset with point labels."""

    def __init__(
        self,
        root_path,
        chromatograms,
        labels=None,
        extra_path=None,
        transform=None,
        preload=False):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram 
                filenames.
            labels (string): Filename of the npy file with labels.
            extra_path (string, optional): Path to folder containing additional
                traces.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        if labels:
            self.labels = np.load(os.path.join(self.root_dir, labels))
        else:
            self.labels = False
        
        self.extra_dir = extra_path
        self.transform = transform
        self.preload = preload

        if self.preload:
            npys = []
            for i in range(len(self.chromatograms)):
                npys.append(self.load_chromatogram(i))
            
            self.chromatogram_npy = np.stack(npys)
    
    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms.iloc[idx, 0]
        
        if self.preload:
            chromatogram = self.chromatogram_npy[chromatogram_id]
        else:
            chromatogram = self.load_chromatogram(idx)

        if isinstance(self.labels, np.ndarray):
            label = self.labels[chromatogram_id].astype(float)
        else:
            label = np.zeros(chromatogram.shape[1])

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label

    def load_npy(self, npy_names):
        npy = np.load(npy_names[0]).astype(float)

        if len(npy_names) > 1:
            for i in range(1, len(npy_names)):
                npy = np.vstack((npy, np.load(npy_names[i]).astype(float)))

        return npy

    def load_chromatogram(self, idx):
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 1]) + '.npy'

        npy_names = [chromatogram_name]

        if self.extra_dir:
            extra_name = os.path.join(
                self.extra_dir,
                self.chromatograms.iloc[idx, 1]) + '_Extra.npy'

            npy_names.append(extra_name)
            
        chromatogram = self.load_npy(npy_names)

        return chromatogram

    def get_bb(self, idx):
        bb_start, bb_end = \
            self.chromatograms.iloc[idx, 2], self.chromatograms.iloc[idx, 3]
        
        return bb_start, bb_end

class ChromatogramsBBoxDataset(Dataset):
    """Whole Chromatograms dataset with bounding boxes."""

    def __init__(self, root_path, chromatograms, transform=None):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
                filenames with bounding boxes.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.transform = transform
    
    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id = self.chromatograms.iloc[idx, 0]
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 1]) + '.npy'
        chromatogram = np.load(chromatogram_name).astype(float)
        bb_start, bb_end = \
            self.chromatograms.iloc[idx, 2], self.chromatograms.iloc[idx, 3]
        bbox = (bb_start, bb_end)

        if self.transform:
            chromatogram, bbox = self.transform((chromatogram, bbox))

        return chromatogram, bbox

class ChromatogramSubsectionsDataset(Dataset):
    """Chromatogram Subsections dataset."""

    def __init__(self, root_path, chromatograms, transform=None):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram filenames
                and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.labels = [self.chromatograms.iloc[idx, 2] for idx in range(
            len(self.chromatograms))]
        self.transform = transform
    
    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_name = os.path.join(
            self.root_dir,
            self.chromatograms.iloc[idx, 1]) + '.npy'
        chromatogram = np.load(chromatogram_name).astype(float)
        label = self.labels[idx]

        if self.transform:
            chromatogram, label = self.transform((chromatogram, label))

        return chromatogram, label

class ChromatogramSubsectionsInMemoryDataset(Dataset):
    """Chromatogram Subsections In Memory dataset."""

    def __init__(self, root_path, chromatograms, chromatograms_npy, transform=None):
        """
        Args:
            root_path (string): Path to the root folder.
            chromatograms (string): Filename of CSV with chromatogram
            subsection counter, chromatogram id, start pos, end pos, and
            subsection label.
            chromatograms_npy (string): Filename of npy file with whole
            chromatograms.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g. padding).
        """
        self.root_dir = root_path
        self.chromatograms = pd.read_csv(os.path.join(self.root_dir,
                                         chromatograms))
        self.labels = [self.chromatograms.iloc[idx, 4] for idx in range(
            len(self.chromatograms))]
        self.chromatograms_npy = np.load(
            os.path.join(self.root_dir, chromatograms_npy) + '.npy')
        self.transform = transform
    
    def __len__(self):
        return len(self.chromatograms)

    def __getitem__(self, idx):
        chromatogram_id, start, end = (
            self.chromatograms.iloc[idx, 1],
            self.chromatograms.iloc[idx, 2],
            self.chromatograms.iloc[idx, 3])
        subsection = self.chromatograms_npy[chromatogram_id][:, start:end]
        label = self.labels[idx]

        if self.transform:
            subsection, label = self.transform((subsection, label))

        return subsection, label
