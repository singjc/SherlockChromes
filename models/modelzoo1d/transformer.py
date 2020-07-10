import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=8, mask=False):
        """
        :param k:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.k = k
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)

        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):

        b, t, k = x.size()
        h = self.heads
        assert k == self.k, f'Input embedding dim ({e}) should match layer embedding dim ({self.k})'

        keys = self.tokeys(x).view(b, t, h, k)
        queries = self.toqueries(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, ff_hidden_mult * k),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * k, k)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
