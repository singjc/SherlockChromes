import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modelzoo1d.dain import DAIN_Layer
from models.modelzoo1d.deeplab_1d import DeepLab1d
from models.modelzoo1d.transformer import TransformerBlock


class BaselineSegmentationNet(nn.Module):
    def __init__(self, output_mode='loc'):
        super(BaselineSegmentationNet, self).__init__()
        self.output_mode = output_mode
        self.convnet = nn.Sequential(
            nn.Conv1d(498, 256, 13, padding=6),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU())
        self.to_logits = nn.Conv1d(8, 1, 1, padding=0)
        self.to_probs = nn.Sigmoid()

    def forward(self, x):
        b, _, length = x.size()
        out = self.convnet(x)
        out_dict = {}
        out_dict['loc'] = out

        out_dict['loc'] = self.to_logits(out)
        attn = torch.sigmoid(out_dict['loc'])
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out_dict['attn'] = attn
        out_dict['cla'] = self.to_logits(
            torch.sum(out * out_dict['attn'], dim=2, keepdim=True))

        for mode in out_dict:
            if mode in ['loc', 'cla']:
                out_dict[mode] = self.to_probs(out_dict[mode])

            if len(out_dict[mode].size()) > 2:
                out_dict[mode] = out_dict[mode].view(b, -1)

        if self.output_mode == 'loc':
            return out_dict['loc']
        elif self.output_mode == 'cla':
            return out_dict['cla']
        elif self.output_mode == 'attn':
            return out_dict['attn']
        elif self.output_mode == 'all':
            return out_dict
        else:
            raise NotImplementedError


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
        use_pos_emb=True,
        aggregator_mode='instance_linear_softmax',
        output_mode='loc'
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

        self.init_encoder = nn.Linear(in_channels, k)
        self.use_pos_emb = use_pos_emb
        self.pos_emb = nn.Embedding(seq_length, k)
        self.pos_array = torch.arange(seq_length)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
        x = x.transpose(1, 2).contiguous()
        x = self.init_encoder(x)
        b, t, k = x.size()

        if self.use_pos_emb:
            # generate position embeddings
            positions = self.pos_emb(
                self.pos_array.to(self.device))[None, :, :].expand(b, t, k)

            x = x + positions

        x = self.t_blocks(x)

        out_dict = {}
        out_dict['attn'] = torch.zeros(b, t, 1)

        if self.aggregator_mode == 'instance_linear_softmax':
            out_dict['loc'] = self.to_logits(x)
            attn = torch.sigmoid(out_dict['loc'])
            out_dict['loc'] = self.to_probs(out_dict['loc'])
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            out_dict['attn'] = attn
            out_dict['cla'] = torch.sum(
                out_dict['loc'] * out_dict['attn'], dim=1)
        elif self.aggregator_mode == 'embed_linear_softmax':
            out_dict['loc'] = self.to_logits(x)
            attn = torch.sigmoid(out_dict['loc'])
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            out_dict['attn'] = attn
            out_dict['cla'] = self.to_logits(
                torch.sum(x * out_dict['attn'], dim=1, keepdim=True))
        else:
            raise NotImplementedError

        if 'embed' in self.aggregator_mode:
            for mode in ['loc', 'cla']:
                out_dict[mode] = self.to_probs(out_dict[mode])

        for mode in out_dict:
            if len(out_dict[mode].size()) > 2:
                out_dict[mode] = out_dict[mode].view(b, -1)

        if self.output_mode == 'loc':
            return out_dict['loc']
        elif self.output_mode == 'cla':
            return out_dict['cla']
        elif self.output_mode == 'attn':
            return out_dict['attn']
        elif self.output_mode == 'all':
            return out_dict
        else:
            raise NotImplementedError
