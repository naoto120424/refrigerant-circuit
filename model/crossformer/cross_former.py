import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from copy import deepcopy

from model.crossformer.cross_encoder import Encoder
from model.crossformer.cross_encoder import Encoder
from model.crossformer.cross_decoder import Decoder
from model.crossformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from model.crossformer.cross_embed import DSW_embedding

from math import ceil


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class SpecEmbedding(nn.Module):
    def __init__(self, dim, num_control_features):
        super().__init__()
        self.spec_emb_list = clones(nn.Linear(1, dim), num_control_features)

    def forward(self, spec):
        spec = torch.unsqueeze(spec, 1)  # bx9 -> bx1x9
        spec = torch.unsqueeze(spec, 1)  # bx1x9 -> bx1x1x9

        for i, spec_embedding in enumerate(self.spec_emb_list):
            if i == 0:
                spec_emb_all = spec_embedding(spec[:, :, :, i])
            else:
                spec_emb_all = torch.cat((spec_emb_all, spec_embedding(spec[:, :, :, i])), dim=1)

        return spec_emb_all


class Crossformer(nn.Module):
    def __init__(self, cfg, args):
        super(Crossformer, self).__init__()
        self.data_dim = cfg.NUM_ALL_FEATURES
        self.dim = args.d_model
        self.depth = args.e_layers
        self.win_size = args.win_size
        self.heads = args.n_heads
        self.in_len = args.in_len
        self.out_len = 1
        self.fc_dim = args.d_ff
        self.seg_len = args.seg_len
        self.merge_win = args.win_size
        self.dropout = args.dropout
        self.factor = args.factor

        # self.baseline = baseline

        # self.device = device

        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, self.dim)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), self.dim))
        self.pre_norm = nn.LayerNorm(self.dim)
        # self.spec_embedding = SpecEmbedding(d_model, self.num_control_features)

        # Encoder
        self.encoder = Encoder(
            self.depth, self.win_size, self.dim, self.heads, self.fc_dim, block_depth=1, dropout=self.dropout, in_seg_num=(self.pad_in_len // self.seg_len), factor=self.factor
        )

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), self.dim))
        self.decoder = Decoder(self.seg_len, self.depth + 1, self.dim, self.heads, self.fc_dim, self.dropout, out_seg_num=(self.pad_out_len // self.seg_len), factor=self.factor)

    def forward(self, x_seq, spec):
        # if (self.baseline):
        #     base = x_seq.mean(dim = 1, keepdim = True)
        # else:
        #     base = 0
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, "b ts_d l d -> (repeat b) ts_d l d", repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        return torch.squeeze(predict_y[:, : self.out_len, 9:], dim=1), None
