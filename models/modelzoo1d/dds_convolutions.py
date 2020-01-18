"""Dynamic Depth Separable Convolutional Neural Networks"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, queries, key, value):
        q_b, qk_c, l = queries.size()
        kv_b, v_c, _ = value.size()
        h = self.heads

        queries = self.to_queries_and_key(queries).view(q_b, h, qk_c, l)
        key = self.to_queries_and_key(key).view(kv_b, h, qk_c, l)
        value = self.to_value(value).view(kv_b, h, v_c, l)

        # Fold heads into the batch dimension
        queries = queries.view(q_b * h, qk_c, l)
        key = key.view(kv_b * h, qk_c, l)
        value = value.view(kv_b * h, v_c, l)

        queries = queries / (qk_c ** (1 / 4))
        key = key / (v_c ** (1 / 4))

        # Get dot product of queries and key, and scale
        dot = torch.matmul(key.transpose(1, 2), queries)
        # dot has size (q_b*h, l, l) containing raw weights

        dot = F.softmax(dot, dim=1)
        # dot now has channel-wise self-attention probabilities

        # Apply the attention to the value
        out = torch.matmul(value, dot).view(q_b, h * v_c, l)

        # Unify heads
        if self.heads > 1:
            out = self.unify_heads(out)

        return out

class DynamicDepthSeparableTimeSeriesTransformerBlock(nn.Module):
    def __init__(
        self,
        c,
        heads,
        depth_multiplier=4,
        dropout=0.0,
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

        self.dropout = nn.Dropout(dropout)
        
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
        heads=1,
        depth_multiplier=4,
        dropout=0.0,
        depth=6,
        kernel_sizes=[3, 15],
        use_template=False,
        cat_template=False):
        super(DDSTSTransformer, self).__init__()
        self.use_template = use_template

        self.cat_template = self.use_template and cat_template

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

        if self.use_template:
            self.template_attn = (
                DynamicDepthSeparableTimeSeriesTemplateAttention(
                    qk_c=transformer_channels,
                    v_c=1,
                    heads=heads,
                    kernel_sizes=kernel_sizes
                )
            )

        if self.cat_template:
            t_out_channels+= 1
        elif self.use_template:
            t_out_channels = 1

        # Maps the final output sequence to class probabilities
        self.to_probs = nn.Sequential(
            DynamicDepthSeparableConv1d(
                t_out_channels,
                1,
                kernel_sizes=kernel_sizes
            ),
            nn.Sigmoid()
        )

    def forward(self, x, template=None, template_label=None):
        b, _, _ = x.size()

        x = self.init_encoder(x)
        x = self.t_blocks(x)

        if self.use_template:
            template = self.init_encoder(template)
            template = self.t_blocks(template)
            x_weighted = self.template_attn(x, template, template_label)

            if self.cat_template:
                x = torch.cat([x, x_weighted], dim=1)
            else:
                x = x_weighted

        x = self.to_probs(x)

        return x.view(b, -1)

class DynamicDepthSeparableConv1dResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[3, 15],
        dilation=1,
        bias=False,
        intermediate_nonlinearity=False
    ):
        super(DynamicDepthSeparableConv1dResBlock, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            DynamicDepthSeparableConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                dilation=dilation,
                bias=bias,
                intermediate_nonlinearity=intermediate_nonlinearity
            )
        )

        self.residual = (
            nn.Identity() if in_channels == out_channels else nn.Conv1d(
                in_channels,
                out_channels,
                1,
                bias=False
            )
        )

    def forward(self, x):
        out = self.network(x) + self.residual(x)

        return out

class EncoderDecoderNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=[64, 32, 8, 4, 1],
        kernel_sizes=[3, 15],
        dilations=[1, 1, 1, 1, 1],
        bias=False,
        intermediate_nonlinearity=False,
        normalize=False,
        freeze=False,
        use_cuda=False
    ):
        super(EncoderDecoderNet, self).__init__()
        self.normalize = normalize

        self.layers = nn.ModuleList()
        
        self.layers.append(
            DynamicDepthSeparableConv1d(
                in_channels=in_channels,
                out_channels=out_channels[0],
                kernel_sizes=kernel_sizes,
                dilation=dilations[0],
                bias=bias,
                intermediate_nonlinearity=intermediate_nonlinearity
            )
        )

        for i in range(1, len(out_channels)):
            self.layers.append(
                DynamicDepthSeparableConv1dResBlock(
                    in_channels=out_channels[i - 1],
                    out_channels=out_channels[i],
                    kernel_sizes=kernel_sizes,
                    dilation=dilations[i],
                    bias=bias,
                    intermediate_nonlinearity=intermediate_nonlinearity
                )
            )

        self.network = nn.Sequential(*self.layers)

        if freeze:
            for param in self.network.parameters():
                param.requires_grad = False

        if use_cuda:
            self.network = self.network.to('cuda:0')

    def forward(self, sequence_batch):
        out = self.network(sequence_batch)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=1, eps=1e-12, out=None)

        return out

class Pyramid(nn.Module):
    def __init__(
        self, 
        in_channels=32,
        out_channels=32,
        kernel_sizes=[3, 15],
        dilations=[1, 2, 6, 12, 18]
    ):
        super(Pyramid, self).__init__()
        self.pyramid = nn.ModuleList()

        self.pyramid.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    1,
                    dilation=dilations[0],
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        )

        for i in range(1, len(dilations)):
            self.pyramid.append(
                nn.Sequential(
                    DynamicDepthSeparableConv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_sizes=kernel_sizes,
                        dilation=dilations[i]
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.global_max_pool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.pyramid[0](x)
        block_size = out.size()

        for layer in self.pyramid[1:]:
            out = torch.cat([out, layer(x)], dim=1)

        out = torch.cat(
            [out, self.global_avg_pool(x).expand(block_size)],
            dim=1
        )

        out = torch.cat(
            [out, self.global_max_pool(x).expand(block_size)],
            dim=1
        )

        return out

class DynamicSegmentationNet(nn.Module):
    def __init__(
        self,
        in_channels=6,
        out_channels=[8, 16, 32, 64, 128],
        kernel_sizes=[3, 15],
        dilations=[1, 1, 1, 1, 1],
        use_unet=False,
        p_dilations=[1, 2, 6, 12, 18],
        depth=1,
        fusion_channels=32,
        heads=1,
        boom_multiplier=4,
        dropout=0.3
    ):
        super(DynamicSegmentationNet, self).__init__()
        self.use_unet = use_unet
        self.use_pyramid = (len(p_dilations) > 0)
        self.use_transformer = (depth > 0)

        # Convolutional Encoder Component
        self.encoder = EncoderDecoderNet(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_sizes=kernel_sizes,
            dilations=dilations
        )

        # Decoder Component
        if self.use_unet:
            self.decoder = EncoderDecoderNet(
                in_channels=out_channels[-1],
                out_channels=out_channels[:-1][::-1] + [fusion_channels],
                kernel_sizes=kernel_sizes,
                dilations=dilations[::-1]
            )
        else:
            self.decoder = nn.Sequential(
                    DynamicDepthSeparableConv1d(
                        in_channels=out_channels[-1],
                        out_channels=fusion_channels,
                        kernel_sizes=kernel_sizes,
                        dilation=1
                    ),
                    nn.BatchNorm1d(fusion_channels),
                    nn.ReLU()
                )

        # Pyramid Component
        if self.use_pyramid:
            self.pyramid = Pyramid(
                in_channels=out_channels[-1],
                out_channels=fusion_channels,
                kernel_sizes=kernel_sizes,
                dilations=p_dilations
            )

        # Transformer Component
        if self.use_transformer:
            self.t_encoder = nn.Conv1d(
                out_channels[-1],
                fusion_channels,
                1,
                bias=False
            )

            self.transformer = []

            for i in range(depth):
                self.transformer.append(
                    DynamicDepthSeparableTimeSeriesTransformerBlock(
                        c=fusion_channels,
                        heads=heads,
                        depth_multiplier=boom_multiplier,
                        dropout=dropout,
                        kernel_sizes=kernel_sizes
                    )
                )

            self.transformer = nn.Sequential(*self.transformer)

        # Maps the final output sequence to class probabilities
        channel_mult = 1

        if self.use_pyramid:
            channel_mult+= (len(p_dilations) + 2)

        if self.use_transformer:
            channel_mult+= 1

        self.to_probs = nn.Sequential(
            nn.Dropout(dropout),
            DynamicDepthSeparableConv1d(
                channel_mult * fusion_channels,
                1,
                kernel_sizes=kernel_sizes
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, _, _ = x.size()

        encoder_out = x

        if self.use_unet:
            skip_outs = []

        for layer in self.encoder.layers[:-1]:
            encoder_out = layer(encoder_out)

            if self.use_unet:
                skip_outs.append(encoder_out)

        encoder_out = self.encoder.layers[-1](encoder_out)
        out = encoder_out

        if self.use_unet:
            for layer in self.decoder.layers[:-1]:
                out = layer(out)
                out = out + skip_outs.pop()

            out = self.decoder.layers[-1](out)
        else:
            out = self.decoder(out)

        if self.use_pyramid:
            out = torch.cat([out, self.pyramid(encoder_out)], dim=1)

        if self.use_transformer:
            t_out = self.t_encoder(encoder_out)
            t_out = self.transformer(t_out)
            out = torch.cat([out, t_out], dim=1)

        out = self.to_probs(out)

        return out.view(b, -1)
