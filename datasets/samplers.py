import numpy as np
import random

def stratified_sampler(data, test_batch_proportion=0.1):
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
