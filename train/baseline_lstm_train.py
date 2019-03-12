import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../models')
sys.path.insert(0, '../datasets')
sys.path.insert(0, '../optimizers')

from baseline_lstm_model import BaselineChromatogramPeakDetector
from chromatograms_dataset import ChromatogramsDataset
from focal_loss import FocalLossBinary
from train import train
from transforms import ToTensor
from visualizer import plot_chromatogram, plot_confusion_matrix

if __name__ == "__main__":  
    # chromatograms = ChromatogramsDataset(
    #     '../../../data/working/ManualValidation',
    #     'chromatograms.csv',
    #     'skyline_exported_labels.npy')

    # model = BaselineChromatogramPeakDetector(6, 64, 1)
    # loss_function = FocalLossBinary()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)

    # with torch.no_grad():
    #     inputs = chromatograms[18][0]
    #     peak_scores = model(inputs)

    #     #Accuracy
    #     peaks = chromatograms[18][1]
    #     output = peak_scores
    #     output = (output > 0.5).float().view(len(peaks))
    #     correct = (output == peaks).float().sum()
    #     print("Accuracy: {:.3f}".format(correct/output.shape[0]))

    # checkpoint_epoch, checkpoint_accuracy = 50, 0
    # epochs = 100
    # for epoch in range(epochs):  
    #     for i in range(len(chromatograms)):
    #     # for i in [18]:
    #         chromatogram, peaks = chromatograms[i][0], chromatograms[i][1]

    #         model.zero_grad()

    #         model.hidden = model.init_hidden()

    #         peak_scores = model(chromatogram)

    #         loss = loss_function(peak_scores.view(len(peaks)), peaks)
    #         loss.backward()
    #         optimizer.step()

    #         #Accuracy
    #         output = peak_scores
    #         output = (output > 0.5).float().view(len(peaks))
    #         correct = (output == peaks).float().sum()
    #         print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(
    #             epoch + 1, epochs, loss.data.item(), correct/output.shape[0]))

    #         if epoch == checkpoint_epoch:
    #             checkpoint_accuracy = correct/output.shape[0]

    # with torch.no_grad():
    #     inputs = chromatograms[18][0]
    #     peak_scores = model(inputs)

    #     #Accuracy
    #     peaks = chromatograms[18][1]
    #     output = peak_scores
    #     output = (output > 0.5).float().view(len(peaks))
    #     correct = (output == peaks).float().sum()
    #     accuracy = correct/output.shape[0]
    #     print("Accuracy: {:.3f}".format(accuracy))
    #     plot_confusion_matrix(peaks, output, classes=['noPeak', 'Peak'], normalize=False,
    #                   title='Normalized confusion matrix')
        
        # if accuracy > checkpoint_accuracy:
        #     torch.save(model.state_dict(), "../../../data/working/overfit_on_chromatogram_18_baseline_lstm.pt")

    data = ChromatogramsDataset(
        '../../../data/working/ManualValidation',
        'chromatograms.csv',
        'skyline_exported_labels.npy', transform=ToTensor())

    model = BaselineChromatogramPeakDetector(6, 64, 1, 32)
    loss_function = FocalLossBinary()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    train(
        data,
        model,
        optimizer,
        loss_function,
        device,
        test_batch_proportion=0.15,
        max_epochs=150,
        train_batch_size=32,
        val_batch_size=32)

    with torch.no_grad():
        inputs = data[0][0]
        peak_scores = model(inputs)

        #Accuracy
        peaks = data[0][1]
        output = peak_scores
        output = (output > 0.5).float().view(len(peaks))
        correct = (output == peaks).float().sum()
        accuracy = correct/output.shape[0]
        print("Accuracy: {:.3f}".format(accuracy))
        plot_confusion_matrix(peaks, output, classes=['noPeak', 'Peak'], normalize=False,
                      title='Normalized confusion matrix')
