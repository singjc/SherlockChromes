import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch

from sklearn.metrics import confusion_matrix

from chromatograms_dataset import ChromatogramsDataset, ChromatogramSubsectionsDataset, ChromatogramSubsectionsInMemoryDataset

def plot_whole_chromatogram(chromatogram, labels=None, bb_start=None, bb_end=None):
    """
    Args:
        chromatogram: (len_of_chromatogram, num_dimensions)
        labels: (len_of_chromatogram, )
    """
    # Convert from single multivariate time series
    # back to separate single variate time series
    # with shape (num_dimensions, len_of_chromatogram)
    if chromatogram.shape[0] == max(chromatogram.shape):
        chromatogram = chromatogram.T

    traces, timepoints, intensities = [], [], []

    for trace in range(len(chromatogram)):
        for timepoint in range(len(chromatogram[trace])):
            traces.append('chromatogram_' + str(trace))
            timepoints.append(timepoint)
            intensities.append(chromatogram[trace][timepoint])

    df = pd.DataFrame(
        {'trace': traces, 'timepoint': timepoints, 'intensity': intensities})

    by_trace = df.groupby('trace')

    for trace, group in by_trace:
        plt.plot(group['timepoint'], group['intensity'], label=trace)

    labels = list((labels >= 0.5))

    for i in range(len(labels)):
        if labels[i] == 1:
            plt.axvline(i, color='r')
            plt.axvline(i + 30, color='r')

    if bb_start and bb_end:
        plt.axvline(bb_start, color='b')
        plt.axvline(bb_end, color='b')

    plt.legend()
    plt.show()

def plot_chromatogram_subsection(chromatogram, labels=None):
    """
    Args:
        chromatogram: (len_of_chromatogram, num_dimensions)
        labels: (len_of_chromatogram, )
    """
    # Convert from single multivariate time series
    # back to separate single variate time series
    # with shape (num_dimensions, len_of_chromatogram)
    if chromatogram.shape[0] == max(chromatogram.shape):
        chromatogram = chromatogram.T

    traces, timepoints, intensities = [], [], []

    for trace in range(len(chromatogram)):
        for timepoint in range(len(chromatogram[trace])):
            traces.append('chromatogram_' + str(trace))
            timepoints.append(timepoint)
            intensities.append(chromatogram[trace][timepoint])

    df = pd.DataFrame(
        {'trace': traces, 'timepoint': timepoints, 'intensity': intensities})

    by_trace = df.groupby('trace')

    for trace, group in by_trace:
        plt.plot(group['timepoint'], group['intensity'], label=trace)

    if labels:
        labels = list(labels)

        peak_start, peak_end = (
            labels.index(1) - 1, len(labels) - labels[::-1].index(1))

        plt.axvline(peak_start, color='b')
        plt.axvline(peak_end, color='b')

    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()

    return ax

def analyze_model(dataset, model, root_dir, idx_filename, mode='val'):
    counter1, counter2, counter3, counter4 = 0, 0, 0, 0
    no_boxes, less_boxes, equal_boxes, more_boxes = [], [], [], []

    idxs = np.loadtxt(os.path.join(root_dir, idx_filename))
    for idx in idxs:
        idx = int(idx)
        chromatogram, labels = dataset[idx]
        dims = chromatogram.shape
        output = model(torch.from_numpy(
                    np.asarray(chromatogram)).view(1, *dims).float())[0]

        num_boxes = (labels == 1).sum()
        num_boxes_predicted = (output > 0.5).sum()
        if num_boxes_predicted == 0:
            counter1+= 1
            no_boxes.append(idx)
        elif 0 < num_boxes_predicted < num_boxes:
            counter2+= 1
            less_boxes.append(idx)
        elif num_boxes_predicted == num_boxes:
            counter3+= 1
            equal_boxes.append(idx)
        elif num_boxes_predicted > num_boxes:
            counter4+= 1
            more_boxes.append(idx)

    np.savetxt(os.path.join(root_dir, mode + '_no_boxes.txt'), np.array(no_boxes))
    np.savetxt(os.path.join(root_dir, mode + '_less_boxes.txt'), np.array(less_boxes))
    np.savetxt(os.path.join(root_dir, mode + '_equal_boxes.txt'), np.array(equal_boxes))
    np.savetxt(os.path.join(root_dir, mode + '_more_boxes.txt'), np.array(more_boxes))

    print(counter1, counter2, counter3, counter4)

def test_model(dataset, model, mode='whole'):
    loop = True
    labels = dataset.labels

    if mode != 'whole':
        positives = np.where(labels == np.int64(1))[0]
        negatives = np.where(labels == np.int64(0))[0]

    while loop:
        retrieve = input('Retrieve: ')

        idx = 0

        if mode != 'whole':
            if retrieve == 'positive':
                idx = np.random.choice(positives)
            elif retrieve == 'negative':
                idx = np.random.choice(negatives)
            else:
                idx = int(float(retrieve))
        else:
            idx = int(float(retrieve))

        chromatogram, true_label = dataset[idx]
        
        print('Chromatogram/subsection id: {}'.format(idx))

        if mode != 'whole':
            print('Chromatogram subsection label: {}'.format(true_label))

        with torch.no_grad():
            dims = chromatogram.shape
            output = model(
                torch.from_numpy(
                    np.asarray(chromatogram)).view(1, *dims).float())[0]

            print(output)

            output = output.numpy()

            if mode != 'whole':
                plot_chromatogram_subsection(chromatogram, output)
            else:
                plot_whole_chromatogram(chromatogram, output, *dataset.get_bb(idx))


if __name__ == "__main__":
    # e.g. ../../../data/working/ManualValidation
    root_dir = input('Dataset root dir: ')
    # e.g. chromatograms.csv
    chromatograms_filename = input('Dataset chromatograms csv: ')
    # e.g. skyline_exported_labels.npy
    labels_filename = input('Dataset labels npy: ')
    dataset = ChromatogramsDataset(
        root_dir,
        chromatograms_filename,
        labels_filename)

    # e.g. ../../../data/output/1dcnn_xcept_whole/1dcnn_model_80_loss=0.01623706.pth
    model_filename = input('Model pth: ')
    model = torch.load(model_filename)
    model.to('cpu')
    model.batch_size = 1
    model.eval()

    if input('Analyze? [yes, no]: ') == 'yes':
        # e.g. ../../../data/output/1dcnn_hybrid_whole
        output_root_dir = input('Output root dir: ')
        # e.g. val_idx.txt
        idx_filename = input('Val Idx filename: ')
        analyze_model(dataset, model, output_root_dir, idx_filename)
        # e.g. test_idx.txt
        idx_filename = input('Test Idx filename: ')
        analyze_model(dataset, model, output_root_dir, idx_filename, mode='test')

    test_model(dataset, model)
