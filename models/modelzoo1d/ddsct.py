"""Dynamic Depth Separable Convolutional Transformer"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modelzoo1d.dain import DAIN_Layer

class DynamicDepthSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3, 15],
        dilation=1,
        bias=False,
        intermediate_nonlinearity=False
    ):
        super(DynamicDepthSeparableConv1d, self).__init__()
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            1,
            bias=bias
        )

        self.intermediate_nonlinearity = intermediate_nonlinearity

        if self.intermediate_nonlinearity:
            self.nonlinear_activation = nn.ReLU()

        # Create dynamic kernel gating mechanism
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)

        self.dynamic_gate = nn.Parameter(
          torch.Tensor([1.0 / self.num_kernels for _ in self.kernel_sizes])
        )

        self.dynamic_depthwise = nn.ModuleList([])
        for kernel_size in self.kernel_sizes:
            conv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=((kernel_size - 1) // 2 * dilation),
                dilation=dilation,
                groups=out_channels,
                bias=bias
            )
            
            self.dynamic_depthwise.append(conv)

    def forward(self, x):
        out = self.pointwise(x)

        if self.intermediate_nonlinearity:
            out = self.nonlinear_activation(out)

        # Apply dynamically sized kernels to values
        dynamic_out = []
        for dynamic_conv in self.dynamic_depthwise:
            dynamic_out.append(dynamic_conv(out))

        out = torch.sum(
            torch.stack(
                dynamic_out,
                dim=-1
            ) * F.softmax(self.dynamic_gate, dim=-1),
            dim=-1
        )

        return out

class DynamicDepthSeparableTimeSeriesSelfAttention(nn.Module):
    def __init__(self, c, heads=8, kernel_sizes=[3, 15]):
        super().__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries_and_keys = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes
        )

        self.to_values = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes
        )

        # This unifies the outputs of the different heads into a single 
        # c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * c, c, 1, bias=False)

    def forward(self, x):
        b, c, l = x.size()
        h = self.heads

        queries = self.to_queries_and_keys(x).view(b, h, c, l)
        keys = self.to_queries_and_keys(x).view(b, h, c, l)
        values = self.to_values(x).view(b, h, c, l)

        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, l)
        keys = keys.view(b * h, c, l)
        values = values.view(b * h, c, l)

        # Get dot product of queries and keys, and scale
        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot now has size (b*h, l, l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, l)

        # Unify heads
        if self.heads > 1:
            out = self.unify_heads(out)

        return out

class DynamicDepthSeparableTimeSeriesTemplateAttention(nn.Module):
    def __init__(self, qk_c, v_c, heads=8, kernel_sizes=[3, 15]):
        super().__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries_and_keys = DynamicDepthSeparableConv1d(
            qk_c,
            qk_c * heads,
            kernel_sizes=kernel_sizes
        )

        self.to_values = DynamicDepthSeparableConv1d(
            v_c,
            v_c * heads,
            kernel_sizes=kernel_sizes
        )

        # This unifies the outputs of the different heads into a single 
        # v_c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * v_c, v_c, 1, bias=False)

    def forward(self, queries, keys, values):
        if len(values.size()) == 2:
            values = values.unsqueeze(1)

        q_b, qk_c, l = queries.size()
        kv_b, v_c, _ = values.size()
        h = self.heads

        queries = self.to_queries_and_keys(queries).view(q_b, h, qk_c, l)
        keys = self.to_queries_and_keys(keys).view(kv_b, h, qk_c, l)
        values = self.to_values(values).view(kv_b, h, v_c, l)

        # Fold heads into the batch dimension
        queries = queries.view(q_b * h, qk_c, l)
        keys = keys.view(kv_b * h, qk_c, l)
        values = values.view(kv_b * h, v_c, l)

        # Get dot product of queries and key, and scale
        queries = queries / (qk_c ** (1 / 4))
        keys = keys / (v_c ** (1 / 4))

        if kv_b > 1:
            dot = torch.matmul(
                keys.transpose(1, 2).contiguous().view(kv_b * h * l, qk_c),
                queries
            ).view(q_b * h, kv_b * h, l, l)
            # dot now has size (q_b*h, kv_b*h, l, l) containing raw weights

            dot = F.softmax(dot, dim=2)
            # dot now has channel-wise self-attention probabilities

            # Apply the attention to the values
            out = torch.matmul(values, dot)
            # out now has size (q_b*h, kv_b*h, v_c, l)

            out = torch.sum(out, dim=1)
            # out now has size (q_b*h, v_c, l)
        else:
            dot = torch.matmul(keys.transpose(1, 2), queries)
            # dot now has size (q_b*h, l, l) containing raw weights

            dot = F.softmax(dot, dim=1)
            # dot now has channel-wise self-attention probabilities

            # Apply the attention to the values
            out = torch.matmul(values, dot)
            # out now has size (q_b*h, v_c, l)

        out = out.view(q_b, h * v_c, l)

        # Unify heads
        if self.heads > 1:
            out = self.unify_heads(out)

        return out, dot

class DynamicDepthSeparableTimeSeriesTransformerBlock(nn.Module):
    def __init__(
        self,
        c,
        heads,
        depth_multiplier=4,
        dropout=0.1,
        kernel_sizes=[3, 15]):
        super().__init__()
        self.attention = DynamicDepthSeparableTimeSeriesSelfAttention(
            c,
            heads=heads,
            kernel_sizes=kernel_sizes
        )

        # Instance norm instead of layer norm
        self.norm1 = nn.InstanceNorm1d(c, affine=True)
        self.norm2 = nn.InstanceNorm1d(c, affine=True)

        # 1D Convolutions instead of FC
        self.feed_forward = nn.Sequential(
            nn.Conv1d(c, depth_multiplier * c, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(depth_multiplier * c, c, 1, bias=False))

        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(self.dropout(attended) + x)

        fed_forward = self.feed_forward(x)
        x = self.norm2(self.dropout(fed_forward) + x)

        return x

class DDSTSTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        transformer_channels=32,
        heads=8,
        depth_multiplier=4,
        dropout=0.1,
        depth=6,
        kernel_sizes=[3, 15],
        normalize=False,
        normalization_mode='full',
        return_normalized=False,
        use_templates=False,
        cat_templates=False,
        return_attn=False,
        probs=True):
        super(DDSTSTransformer, self).__init__()
        self.normalize = normalize
        self.return_normalized = self.normalize and return_normalized
        self.use_templates = use_templates
        self.cat_templates = self.use_templates and cat_templates
        self.return_attn = self.use_templates and return_attn
        self.probs = probs

        if self.normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=in_channels
            )

        self.init_encoder = nn.Conv1d(
            in_channels,
            transformer_channels,
            1,
            bias=False
        )

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        t_blocks = []
        for i in range(depth):
            t_blocks.append(
                DynamicDepthSeparableTimeSeriesTransformerBlock(
                    c=transformer_channels,
                    heads=heads,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout,
                    kernel_sizes=kernel_sizes
                )
            )
        self.t_blocks = nn.Sequential(*t_blocks)

        t_out_channels = transformer_channels

        if self.use_templates:
            self.templates_attn = (
                DynamicDepthSeparableTimeSeriesTemplateAttention(
                    qk_c=transformer_channels,
                    v_c=1,
                    heads=heads,
                    kernel_sizes=kernel_sizes
                )
            )

        if self.cat_templates:
            t_out_channels+= 1
        elif self.use_templates:
            t_out_channels = 1

        # Maps the final output sequence to class probabilities
        self.to_logits = nn.Sequential(
            DynamicDepthSeparableConv1d(
                t_out_channels,
                1,
                kernel_sizes=[1, 3]
            )
        )

        self.to_probs = nn.Sigmoid()

    def forward(self, x, templates=None, templates_label=None):
        b, _, _ = x.size()

        if self.normalize:
            x = self.normalization_layer(x)

        out = self.init_encoder(x)
        out = self.t_blocks(out)

        if self.use_templates:
            if self.normalize:
                templates = self.normalization_layer(templates)

            templates = self.init_encoder(templates)
            templates = self.t_blocks(templates)
            out_weighted, attn_matrix = self.templates_attn(
                out, templates, templates_label)

            if self.cat_templates:
                out = torch.cat([out, out_weighted], dim=1)
            else:
                out = out_weighted

        out = self.to_logits(out)

        if self.probs:
            out = self.to_probs(out)

        if self.return_normalized or self.return_attn:
            all_outs = [out.view(b, -1)]

        if self.return_normalized:
            all_outs.append(x)
        else:
            all_outs.append(None)
        
        if self.return_attn:
            all_outs.append(attn_matrix)
        else:
            all_outs.append(None)

        if self.return_normalized or self.return_attn:
            return all_outs
        return out.view(b, -1)