import torch
import torch.nn as nn

class BaselineChromatogramPeakDetector(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_size):
        super(BaselineChromatogramPeakDetector, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)

        self.hidden2output = nn.Linear(hidden_dim * 2, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim),
                torch.zeros(2, 1, self.hidden_dim))

    def forward(self, chromatogram):
        lstm_out, self.hidden = self.lstm(
            chromatogram.view(len(chromatogram), 1, -1), self.hidden)
        peak_space = self.hidden2output(lstm_out.view(len(chromatogram), -1))
        peak_scores = torch.sigmoid(peak_space)

        return peak_scores


if __name__ == "__main__":
    pass
