import numpy as np
import random

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
