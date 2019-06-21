import numpy as np
import os
import pandas as pd
import random

from collections import defaultdict
from itertools import chain

class StratifiedSampler(object):
    """Sample through class stratified strategy."""

    def __call__(self, data, test_batch_proportion=0.1):
        labels = data.labels

        train_idx, val_idx, test_idx = [], [], []

        for i in np.unique(labels):
            idx = list(np.where(labels == i)[0])
            random.shuffle(idx)

            n = len(idx)
            n_test = int(n * test_batch_proportion)
            n_train = n - 2 * n_test

            train_idx+= idx[:n_train]
            val_idx+= idx[n_train:(n_train + n_test)]
            test_idx+= idx[(n_train + n_test):]

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx

class StratifiedSubSampler(object):
    """Subsample in stratified manner such that there are equal amounts of
    positive and negative labelled chromatograms."""

    def __call__(self, data, test_batch_proportion=0.1):
        labels = data.labels

        train_idx, val_idx, test_idx = [], [], []

        n_test_positive, n_train_positive = 0, 0

        for i in [np.int64(1), np.int64(0)]:
            idx = list(np.where(labels == i)[0])
            random.shuffle(idx)

            if i == np.int64(1):
                n = len(idx)
                n_test = int(n * test_batch_proportion)
                n_train = n - 2 * n_test
                n_test_positive = n_test
                n_train_positive = n_train
            else:
                n_test = n_test_positive
                n_train = n_train_positive

            train_idx+= idx[:n_train]
            val_idx+= idx[n_train:(n_train + n_test)]
            test_idx+= idx[(n_train + n_test):]

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx

class LoadingSampler(object):
    """Load pre-existing idx numpy txt files."""

    def __init__(self, **kwargs):
        self.shuffle = kwargs['shuffle']
        self.train_idx_filename = os.path.join(
            kwargs['root_path'], kwargs['train_idx_filename'])
        self.val_idx_filename = os.path.join(
            kwargs['root_path'], kwargs['val_idx_filename'])
        self.test_idx_filename = os.path.join(
            kwargs['root_path'], kwargs['test_idx_filename'])

    def __call__(self, data=None, test_batch_proportion=None):
        train_idx = [int(i) for i in np.loadtxt(self.train_idx_filename)]
        val_idx = [int(i) for i in np.loadtxt(self.val_idx_filename)]
        test_idx = [int(i) for i in np.loadtxt(self.test_idx_filename)]

        if self.shuffle:
            random.shuffle(train_idx)
            random.shuffle(val_idx)
            random.shuffle(test_idx)

        return train_idx, val_idx, test_idx

class GroupBySequenceSampler(object):
    """Sample by sequence groups. 
    Naked sequences are sequences without modifications."""

    def __init__(self, **kwargs):
        self.group_size = kwargs['group_size']
        self.naked = kwargs['naked']

    def __call__(self, data, test_batch_proportion=0.1):
        seq_to_idx = defaultdict(list)

        for i in range(len(data)):
            seq = data.chromatograms.iloc[i, 1].split('_')[-2]

            if self.naked:
                mod_start = seq.rfind('[')
                mod_end = seq.find(']')

                if mod_start != -1 and mod_end != -1:
                    seq = seq[:mod_start] + seq[mod_end + 1:]

            seq_to_idx[seq].append(i)

        grouped_idx = [seq_to_idx[seq] for seq in seq_to_idx]

        random.shuffle(grouped_idx)

        n = int(len(data) / self.group_size)
        n_test = int(n * test_batch_proportion)
        n_train = n - 2 * n_test

        train_idx = list(chain.from_iterable(grouped_idx[:n_train]))
        val_idx = list(
            chain.from_iterable(grouped_idx[n_train:(n_train + n_test)]))
        test_idx = list(chain.from_iterable(grouped_idx[(n_train + n_test):]))

        random.shuffle(train_idx)
        random.shuffle(val_idx)
        random.shuffle(test_idx)

        return train_idx, val_idx, test_idx
