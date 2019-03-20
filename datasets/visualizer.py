import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from chromatograms_dataset import ChromatogramsDataset, ChromatogramSubsectionsDataset

def plot_chromatogram(chromatogram, labels=None):
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

        plt.axvline(peak_start, color='r')
        plt.axvline(peak_end, color='r')

    for i in range(0, len(chromatogram[0]), 10):
        plt.axvline(i, color='b')

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


if __name__ == "__main__":
    # chromatograms = ChromatogramsDataset(
    #     '../../../data/working/ManualValidation',
    #     'chromatograms.csv',
    #     'skyline_exported_labels.npy')
    # plot_chromatogram(chromatograms[0][0], chromatograms[0][1])

    chromatograms = ChromatogramSubsectionsDataset(
        '../../../data/working/ManualValidationSliced_20_1',
        'chromatograms.csv')

    for i in range(len(chromatograms)):
        print(i)
        chromatogram, label = chromatograms[i]

        if label == 1:
            plot_chromatogram(chromatogram)
