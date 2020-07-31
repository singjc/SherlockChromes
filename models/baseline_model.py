import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modelzoo1d.dain import DAIN_Layer
from models.modelzoo1d.deeplab_1d import DeepLab1d
from models.modelzoo1d.transformer import TransformerBlock


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


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        k=32,
        heads=8,
        depth=6,
        seq_length=175,
        normalize=False,
        normalization_mode='full',
        aggregator_mode='max_pool',
        output_mode='strong'
    ):
        super(BaselineTransformer, self).__init__()
        self.aggregator_mode = aggregator_mode
        self.output_mode = output_mode

        if normalize:
            self.normalization_layer = DAIN_Layer(
                mode=normalization_mode,
                input_dim=in_channels)
        else:
            self.normalization_layer = nn.Identity()

        self.init_encoder = nn.Conv1d(in_channels, k, 1)
        self.pos_emb = nn.Embedding(seq_length, k)

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.pos_array = torch.arange(seq_length).to(device)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        t_blocks = []
        for i in range(depth):
            t_blocks.append(TransformerBlock(k=k, heads=heads))
        self.t_blocks = nn.Sequential(*t_blocks)

        # Maps the final output sequence to class logits
        self.to_logits = nn.Linear(k, 1)
        self.to_probs = nn.Sigmoid()

    def forward(self, x):
        x = self.normalization_layer(x)
        x = self.init_encoder(x)
        x = x.transpose(1, 2).contiguous()
        b, t, k = x.size()

        # generate position embeddings
        positions = self.pos_emb(self.pos_array)[None, :, :].expand(b, t, k)

        x = x + positions
        x = self.t_blocks(x)

        out_dict = {}

        out_dict['strong'] = self.to_logits(x)

        if self.aggregator_mode == 'max_pool':
            out_dict['weak'] = self.to_logits(x.max(dim=1)[0])
        elif self.aggregator_mode == 'avg_pool':
            out_dict['weak'] = self.to_logits(x.mean(dim=1))
        elif self.aggregator_mode == 'attn_pool':
            attn_mask = F.softmax(out_dict['strong'], dim=1)
            out_dict['strong'] = self.to_probs(out_dict['strong'])
            out_dict['weak'] = (
                torch.sum(
                    out_dict['strong']
                    * attn_mask, dim=1)
                / torch.sum(attn_mask, dim=1))
        else:
            raise NotImplementedError

        if self.aggregator_mode != 'attn_pool':
            for mode in out_dict:
                out_dict[mode] = self.to_probs(out_dict[mode])

        out_dict['strong'] = out_dict['strong'].view(b, -1)

        if self.output_mode == 'strong':
            return out_dict['strong']
        elif self.output_mode == 'weak':
            return out_dict['weak']
        elif self.output_mode == 'both':
            return out_dict
        else:
            raise NotImplementedError
