import torch
import torch.nn as nn

from .modelzoo1d.transformer import TransformerBlock

class BaselineSegmentationNet(nn.Module):
    def __init__(self):
        super(BaselineSegmentationNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(29, 64, 11, padding=5),
                                     nn.BatchNorm1d(64),
                                     nn.ReLU(),
                                     nn.Conv1d(64, 32, 9, padding=4),
                                     nn.BatchNorm1d(32),
                                     nn.ReLU(),
                                     nn.Conv1d(32, 16, 7, padding=3),
                                     nn.BatchNorm1d(16),
                                     nn.ReLU(),
                                     nn.Conv1d(16, 1, 3, padding=1),
                                     nn.BatchNorm1d(1),
                                     nn.Sigmoid())

    def forward(self, x):
        output = self.convnet(x)

        return output

class BaselineEmbeddingNet(nn.Module):
    def __init__(self):
        super(BaselineEmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv1d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2),
                                     nn.Conv1d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool1d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 593, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

class BaselineTripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(BaselineTripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x_1, x_2, x_3):
        output_1 = self.embedding_net(x_1)
        output_2 = self.embedding_net(x_2)
        output_3 = self.embedding_net(x_3)

        return output_1, output_2, output_3

    def get_embedding(self, x):
        return self.embedding_net(x)

class BaselineTransformer(nn.Module):
    def __init__(self, in_channels, k, heads, depth, seq_length):
        super().__init__()
        self.init_encoder = nn.Conv1d(in_channels, k, 1)

        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Sequential(
            nn.Linear(k, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing 
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the 
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        x = self.init_encoder(x)
        x = x.transpose(1, 2).contiguous()
        b, t, k = x.size()

        # generate position embeddings
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = x + positions
        x = self.tblocks(x)

        x = self.toprobs(x)

        x = x.transpose(1, 2).contiguous()

        return x