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
        intermediate_nonlinearity=False,
        pad=True
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

        self.dynamic_depthwise = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            padding = ((kernel_size - 1) // 2 * dilation) if pad else 0
            conv = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias
            )
            self.dynamic_depthwise.append(conv)

    def get_gate(self):
        return self.dynamic_gate

    def forward(self, x):
        out = self.pointwise(x)

        if self.intermediate_nonlinearity:
            out = self.nonlinear_activation(out)

        # Apply dynamically sized kernels to values
        dynamic_out = []
        for dynamic_conv in self.dynamic_depthwise:
            dynamic_out.append(dynamic_conv(out))

        # Fallback to normal depth separable convolution if single kernel size
        if self.num_kernels > 1:
            out = torch.sum(
                torch.stack(
                    dynamic_out,
                    dim=-1
                ) * F.softmax(self.dynamic_gate, dim=-1),
                dim=-1
            )
        else:
            out = dynamic_out[0]

        return out

class DynamicDepthSeparableConv1dMultiheadAttention(nn.Module):
    def __init__(
        self,
        c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False):
        super(DynamicDepthSeparableConv1dMultiheadAttention, self).__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes
        self.save_attn = save_attn

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries = DynamicDepthSeparableConv1d(
            c,
            c * heads,
            kernel_sizes=kernel_sizes
        )

        if share_encoder:
            self.to_keys = self.to_queries
        else:
            self.to_keys = DynamicDepthSeparableConv1d(
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
            self.unify_heads = nn.Conv1d(heads * c, c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None

    def get_attn(self):
        return self.attn

    def forward(self, q, k=None, v=None):
        b, c, q_l = q.size()
        h = self.heads

        if k is None:
            k = q
            k_l = q_l
        else:
            _, _, k_l = k.size()

        if v is None:
            v = q
            v_l = q_l
        else:
            _, _, v_l = k.size()

        assert k_l == v_l, 'key and value length must be equal'

        queries = self.to_queries(q).view(b, h, c, q_l)
        keys = self.to_keys(k).view(b, h, c, k_l)
        values = self.to_values(v).view(b, h, c, v_l)

        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, q_l)
        keys = keys.view(b * h, c, k_l)
        values = values.view(b * h, c, v_l)

        # Get dot product of queries and keys, and scale
        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot now has size (b*h, k_l, q_l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, q_l)

        # Unify heads
        out = self.unify_heads(out)

        if self.save_attn:
            self.attn = dot

        return out

class DynamicDepthSeparableConv1dTemplateAttention(nn.Module):
    def __init__(
        self,
        qk_c,
        v_c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False):
        super(
            DynamicDepthSeparableConv1dTemplateAttention, self).__init__()
        self.heads = heads
        self.kernel_sizes = kernel_sizes
        self.save_attn = save_attn

        # These compute the queries, keys, and values for all 
        # heads (as a single concatenated vector)
        self.to_queries = DynamicDepthSeparableConv1d(
            qk_c,
            qk_c * heads,
            kernel_sizes=kernel_sizes
        )

        if share_encoder:
            self.to_keys = self.to_queries
        else:
            self.to_keys = DynamicDepthSeparableConv1d(
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
            self.unify_heads = nn.Conv1d(heads * v_c, v_c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None

    def get_attn(self):
        return self.attn

    def forward(self, queries, keys, values):
        if len(values.size()) == 2:
            values = values.unsqueeze(1)

        q_b, qk_c, l = queries.size()
        kv_b, v_c, _ = values.size()
        h = self.heads

        queries = self.to_queries(queries).view(q_b, h, qk_c, l)
        keys = self.to_keys(keys).view(kv_b, h, qk_c, l)
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
                keys.transpose(1, 2).contiguous().view(kv_b * h, l, qk_c),
                queries
            )
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
        out = self.unify_heads(out)

        if self.save_attn:
            self.attn = dot

        return out

class Conv1dFeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Conv1dFeedForwardNetwork, self).__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Conv1d(
                n,
                k,
                1
            ) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x

class DynamicDepthSeparableConv1dTransformerBlock(nn.Module):
    def __init__(
        self,
        c,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False,
        depth_multiplier=4,
        dropout=0.1):
        super(DynamicDepthSeparableConv1dTransformerBlock, self).__init__()
        self.attention = DynamicDepthSeparableConv1dMultiheadAttention(
            c,
            heads=heads,
            kernel_sizes=kernel_sizes,
            share_encoder=share_encoder,
            save_attn=save_attn
        )

        # Instance norm instead of layer norm
        self.norm1 = nn.InstanceNorm1d(c, affine=True)
        self.norm2 = nn.InstanceNorm1d(c, affine=True)

        # 1D Convolutions instead of FC
        self.feed_forward = Conv1dFeedForwardNetwork(
            c,
            depth_multiplier * c,
            c,
            num_layers=2
        )

        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, q, k=None, v=None):
        out = q
        attended = self.attention(q, k, v)
        out = self.norm1(self.dropout(attended) + out)
        fed_forward = self.feed_forward(out)
        out = self.norm2(self.dropout(fed_forward) + out)

        return out

class TimeSeriesAttentionPooling(nn.Module):
    def __init__(
        self,
        num_queries,
        c,
        heads=1,
        save_attn=False):
        super(TimeSeriesAttentionPooling, self).__init__()
        self.num_queries = num_queries
        self.heads = heads
        self.save_attn = save_attn

        # This represents the queries for the weak binary global label
        self.query_embeds = nn.Embedding(self.num_queries, c)

        # These compute the queries for all heads as a single concatenated
        # vector
        if self.heads > 1:
            self.to_queries = nn.Conv1d(
                c,
                c * heads,
                1,
                bias=False
            )
        else:
            self.to_queries = nn.Identity()

        # This unifies the outputs of the different heads into a single 
        # c-vector
        if self.heads > 1:
            self.unify_heads = nn.Conv1d(heads * c, c, 1)
        else:
            self.unify_heads = nn.Identity()

        self.attn = None

    def get_attn(self):
        return self.attn

    def forward(self, x):
        b, c, l = x.size()
        h = self.heads
        queries = self.to_queries(
            self.query_embeds.weight
            .transpose(0, 1)
            .unsqueeze(0)
            .repeat(b, 1, 1)
        )
        keys = values = x.repeat(h, 1, 1)
        
        # Fold heads into the batch dimension
        queries = queries.view(b * h, c, self.num_queries)

        # Get dot product of queries and keys, and scale
        queries = queries / (c ** (1 / 4))
        keys = keys / (c ** (1 / 4))

        dot = torch.bmm(keys.transpose(1, 2), queries)
        # dot now has size (b*h, l, num_queries) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the self attention to the values
        out = torch.bmm(values, dot).view(b, h * c, self.num_queries)

        out = self.unify_heads(out)
        # out now has size (b, c, num_queries)

        if self.save_attn:
            self.attn = dot

        return out

class DDSCTransformer(nn.Module):
    def __init__(
        self,
        in_channels=6,
        normalize=False,
        normalization_mode='full',
        save_normalized=False,
        transformer_channels=32,
        depth=6,
        heads=8,
        kernel_sizes=[3, 15],
        share_encoder=False,
        save_attn=False,
        depth_multiplier=4,
        dropout=0.1,
        use_templates=False,
        cat_templates=False,
        aggregator_num_classes=1,
        aggregator_num_heads=1,
        output_num_layers=1,
        output_mode='strong',
        probs=True):
        super(DDSCTransformer, self).__init__()
        self.save_normalized = save_normalized
        self.use_templates = use_templates
        self.cat_templates = self.use_templates and cat_templates
        self.output_mode = output_mode
        self.probs = probs

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=in_channels
            )
        else:
            self.normalization_layer = nn.Identity()

        self.init_encoder = nn.Conv1d(
            in_channels,
            transformer_channels,
            1
        )

        # The sequence of transformer blocks that does all the 
        # heavy lifting
        t_blocks = []
        for i in range(depth):
            t_blocks.append(
                DynamicDepthSeparableConv1dTransformerBlock(
                    transformer_channels,
                    heads=heads,
                    kernel_sizes=kernel_sizes,
                    share_encoder=share_encoder,
                    save_attn=save_attn,
                    depth_multiplier=depth_multiplier,
                    dropout=dropout
                )
            )
        self.t_blocks = nn.Sequential(*t_blocks)

        t_out_channels = transformer_channels

        if self.use_templates:
            self.templates_attn = (
                DynamicDepthSeparableConv1dTemplateAttention(
                    qk_c=transformer_channels,
                    v_c=1,
                    heads=heads,
                    kernel_sizes=kernel_sizes,
                    share_encoder=share_encoder,
                    save_attn=save_attn
                )
            )

        if self.cat_templates:
            t_out_channels+= 1
        elif self.use_templates:
            t_out_channels = 1

        # Aggregates output to single value per batch item
        self.output_aggregator = TimeSeriesAttentionPooling(
            num_queries=aggregator_num_classes,
            c=t_out_channels,
            heads=aggregator_num_heads,
            save_attn=save_attn
        )

        # Maps the final output state(s) to logits
        self.to_logits = Conv1dFeedForwardNetwork(
            t_out_channels,
            t_out_channels,
            1,
            num_layers=output_num_layers
        )
        
        # Maps the final logits to probabilities
        self.to_probs = nn.Sigmoid()

        self.normalized = None

    def get_normalized(self):
        return self.normalized

    def set_output_probs(self, val):
        self.probs = val

    def forward(self, x, templates=None, templates_label=None):
        b, _, _ = x.size()
        x = self.normalization_layer(x)

        if self.save_normalized:
            self.normalized = x

        out = self.init_encoder(x)
        out = self.t_blocks(out)

        if self.use_templates:
            templates = self.normalization_layer(templates)
            templates = self.init_encoder(templates)
            templates = self.t_blocks(templates)
            out_weighted = self.templates_attn(out, templates, templates_label)

            if self.cat_templates:
                out = torch.cat([out, out_weighted], dim=1)
            else:
                out = out_weighted
        
        out_dict = {}

        if self.output_mode == 'weak' or self.output_mode == 'both':
            out_dict['weak'] = self.to_logits(self.output_aggregator(out))

        if self.output_mode == 'strong' or self.output_mode == 'both':
            out_dict['strong'] = self.to_logits(out)

        if self.probs:
            for mode in out_dict:
                out_dict[mode] = self.to_probs(out_dict[mode])

        for mode in out_dict:
            out_dict[mode] = out_dict[mode].view(b, -1)

        if self.output_mode == 'weak':
            return out_dict['weak']
        elif self.output_mode == 'strong':
            return out_dict['strong']
        elif self.output_mode == 'both':
            return out_dict
