import torch
import torch.nn.functional as F

from torch import nn

from .depth_separable_conv_1d import DepthSeparableConv1d

class TimeSeriesSelfAttention(nn.Module):
    def __init__(self, c, heads=8):
        super().__init__()
        self.c, self.heads = c, heads

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_keys = DepthSeparableConv1d(c, c * heads)
        self.to_queries = DepthSeparableConv1d(c, c * heads)
        self.to_values  = DepthSeparableConv1d(c, c * heads)

        # This unifies the outputs of the different heads into 
        # a single k-vector
        self.unify_heads = DepthSeparableConv1d(heads * c, c)

    def forward(self, x):
        b, c, l = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, h, c, l)
        keys = self.to_keys(x).view(b, h, c, l)
        values = self.to_values(x).view(b, h, c, l)

        # - fold heads into the batch dimension
        keys = keys.view(b * h, c, l)
        queries = queries.view(b * h, c, l)
        values = values.view(b * h, c, l)

        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(keys.transpose(1, 2), queries)
        # - dot has size (b*h, l, l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h, c, l)

        # unify heads
        out = out.view(b, h * c, l)

        return self.unify_heads(out)

class TimeSeriesTransformerBlock(nn.Module):
  def __init__(self, c, heads, depth_multiplier=4, dropout=0.0):
    super().__init__()

    self.attention = TimeSeriesSelfAttention(c, heads=heads)

    # Instance norm instead of layer norm
    self.norm1 = nn.InstanceNorm1d(c, affine=True)
    self.norm2 = nn.InstanceNorm1d(c, affine=True)

    # 1D Convolutions instead of FC
    self.feed_forward = nn.Sequential(
      DepthSeparableConv1d(c, depth_multiplier * c),
      nn.ReLU(),
      DepthSeparableConv1d(depth_multiplier * c, c))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    attended = self.attention(x)
    x = self.norm1(attended + x)
    x = self.dropout(x)
    
    fed_forward = self.feed_forward(x)
    x = self.norm2(fed_forward + x)
    x = self.dropout(x)

    return x
