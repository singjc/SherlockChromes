import torch
import torch.nn.functional as F

from torch import nn

from .depth_separable_conv_1d import DepthSeparableConv1d

class TimeSeriesSelfAttention(nn.Module):
    def __init__(self, c, heads=8):
        super().__init__()
        self.heads = heads

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries = DepthSeparableConv1d(c, c * heads)
        self.to_keys = DepthSeparableConv1d(c, c * heads)
        self.to_values  = DepthSeparableConv1d(c, c * heads)

        # This unifies the outputs of the different heads into 
        # a single k-vector
        self.unify_heads = nn.Conv1d(heads * c, c, kernel_size=1)

    def forward(self, x):
        b, c, l = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, h, c, l)
        keys = self.to_keys(x).view(b, h, c, l)
        values = self.to_values(x).view(b, h, c, l)

        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, l)
        keys = keys.view(b * h, c, l)
        values = values.view(b * h, c, l)

        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        # Get dot product of queries and keys, and scale
        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot has size (b*h, l, l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, l)

        # Unify heads
        return self.unify_heads(out)

class DynamicTimeSeriesSelfAttention(nn.Module):
    def __init__(self, c, heads=8, kernel_sizes=[3, 15]):
        super().__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes

        self.num_kernels = len(kernel_sizes)

        # Create dynamic kernel gating mechanism
        self.dynamic_gate = nn.Parameter(
          torch.Tensor([1.0 / self.num_kernels for _ in self.kernel_sizes])
        )

        self.dynamic_convs = nn.ModuleList([])
        for kernel_size in self.kernel_sizes:
            conv = DepthSeparableConv1d(
                c,
                c,
                kernel_size=kernel_size,
                padding=((kernel_size - 1) // 2))

            self.dynamic_convs.append(conv)

        dim_mult = 2 if self.num_kernels > 0 else 1

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries = DepthSeparableConv1d(c, c * heads)
        self.to_keys = DepthSeparableConv1d(c, c * heads)
        self.to_values  = DepthSeparableConv1d(c, c * heads)

        # This unifies the outputs of the different heads and dynamic
        # convolutions into a single c-vector
        self.unify_heads = nn.Conv1d(heads * c * dim_mult, c, kernel_size=1)

    def forward(self, x):
        b, c, l = x.size()
        h = self.heads

        queries = self.to_queries(x).view(b, h, c, l)
        keys = self.to_keys(x).view(b, h, c, l)
        values = self.to_values(x).view(b, h, c, l)

        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, l)
        keys = keys.view(b * h, c, l)
        values = values.view(b * h, c, l)

        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        # Get dot product of queries and keys, and scale
        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot has size (b*h, l, l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, l)

        # Apply dynamically sized kernels to values
        dynamic_out = []
        for dynamic_conv in self.dynamic_convs:
            dynamic_out.append(dynamic_conv(values))

        if dynamic_out:
            dynamic_out = torch.sum(
            torch.stack(
                dynamic_out,
                dim=-1
            ) * F.softmax(self.dynamic_gate, dim=-1),
            dim=-1
            ).view(b, h * c, l)

            out = torch.cat([out, dynamic_out], dim=1)

        # Unify heads and dynamic output
        return self.unify_heads(out)

class TimeSeriesTransformerBlock(nn.Module):
    def __init__(self, c, heads, depth_multiplier=4, dropout=0.0):
        super().__init__()
        self.attention = TimeSeriesSelfAttention(c, heads=heads)

        # Instance norm instead of layer norm
        self.norm1 = nn.InstanceNorm1d(c, affine=True)
        self.norm2 = nn.InstanceNorm1d(c, affine=True)

        # 1D Convolutions instead of FC
        self.projection = nn.Sequential(
            DepthSeparableConv1d(c, depth_multiplier * c),
            nn.ReLU(),
            DepthSeparableConv1d(depth_multiplier * c, c))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)
        
        projected = self.projection(x)
        x = self.norm2(projected + x)
        x = self.dropout(x)

        return x

class DynamicTimeSeriesTransformerBlock(nn.Module):
    def __init__(
        self,
        c,
        heads,
        depth_multiplier=4,
        dropout=0.0,
        kernel_sizes=[3, 15]):
        super().__init__()

        self.attention = DynamicTimeSeriesSelfAttention(
            c,
            heads=heads,
            kernel_sizes=kernel_sizes
        )

        # Instance norm instead of layer norm
        self.norm1 = nn.InstanceNorm1d(c, affine=True)
        self.norm2 = nn.InstanceNorm1d(c, affine=True)

        # 1D Convolutions instead of FC
        self.projection = nn.Sequential(
            DepthSeparableConv1d(c, depth_multiplier * c),
            nn.ReLU(),
            DepthSeparableConv1d(depth_multiplier * c, c))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)
        
        projected = self.projection(x)
        x = self.norm2(projected + x)
        x = self.dropout(x)

        return x
  