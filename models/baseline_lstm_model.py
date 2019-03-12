import numpy as np
import torch
import torch.nn as nn

class BaselineChromatogramPeakDetector(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size, batch_size):
        super(BaselineChromatogramPeakDetector, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

        self.batch_size = batch_size

        self.hidden2output = nn.Linear(
            self.batch_size * hidden_dim * 2,
            self.batch_size * output_size)

        self.hidden, self.cell = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2, self.batch_size, self.hidden_dim),
                torch.zeros(2, self.batch_size, self.hidden_dim))

    def forward(self, chromatogram):
        num_timesteps = np.max(chromatogram.size())

        lstm_out, hidden_temp = self.lstm(
            chromatogram.view(num_timesteps, 1, -1), (self.hidden, self.cell))
        self.hidden = hidden_temp[0]
        self.cell = hidden_temp[1]
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()

        peak_space = self.hidden2output(lstm_out.view(num_timesteps, -1))
        peak_scores = torch.sigmoid(peak_space).view(peak_space.size()[1], peak_space.size()[0])

        return peak_scores


if __name__ == "__main__":
    pass
