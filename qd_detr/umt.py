# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.
"""UMT fusion modules for integrating video and audio features."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """Simple feed forward network"""

    def __init__(self, dims: int, ratio: int = 4, p: float = 0.1):
        super().__init__()
        hidden = int(dims * ratio)
        self.fc1 = nn.Linear(dims, hidden)
        self.fc2 = nn.Linear(hidden, dims)
        self.dropout = nn.Dropout(p)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, dims: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dims)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dims, 2).float() * -(math.log(10000.0) / dims))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BottleneckTransformerLayer(nn.Module):
    """Single bottleneck transformer layer"""

    def __init__(self, dims: int, heads: int = 8, ratio: int = 4, p: float = 0.1):
        super().__init__()
        self.att1 = nn.MultiheadAttention(dims, heads, dropout=p)
        self.att2 = nn.MultiheadAttention(dims, heads, dropout=p)
        self.att3 = nn.MultiheadAttention(dims, heads, dropout=p)
        self.att4 = nn.MultiheadAttention(dims, heads, dropout=p)

        self.ffn1 = FeedForwardNetwork(dims, ratio, p)
        self.ffn2 = FeedForwardNetwork(dims, ratio, p)

        self.norm1 = nn.LayerNorm(dims)
        self.norm2 = nn.LayerNorm(dims)
        self.norm3 = nn.LayerNorm(dims)
        self.norm4 = nn.LayerNorm(dims)
        self.norm5 = nn.LayerNorm(dims)
        self.norm6 = nn.LayerNorm(dims)

    def forward(self, a, b, t, pe=None, mask=None):
        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)

        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0

        dt_q = dt.permute(1, 0, 2)
        ka_k = ka.permute(1, 0, 2)
        da_v = da.permute(1, 0, 2)
        at, _ = self.att1(dt_q, ka_k, da_v, key_padding_mask=key_padding_mask)
        at = at.permute(1, 0, 2)

        kb_k = kb.permute(1, 0, 2)
        db_v = db.permute(1, 0, 2)
        bt, _ = self.att2(dt_q, kb_k, db_v, key_padding_mask=key_padding_mask)
        bt = bt.permute(1, 0, 2)

        t = t + at + bt
        dt = self.norm4(t)

        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        qa_q = qa.permute(1, 0, 2)
        dt_k = dt.permute(1, 0, 2)
        a_att, _ = self.att3(qa_q, dt_k, dt_k)
        a_att = a_att.permute(1, 0, 2)
        a = a + a_att

        qb_q = qb.permute(1, 0, 2)
        b_att, _ = self.att4(qb_q, dt_k, dt_k)
        b_att = b_att.permute(1, 0, 2)
        b = b + b_att

        da = self.norm5(a)
        db = self.norm6(b)
        a = a + self.ffn1(da)
        b = b + self.ffn2(db)
        return a, b, t


class BottleneckTransformer(nn.Module):
    """Stack of bottleneck transformer layers with learnable tokens"""

    def __init__(self, dims: int, num_tokens: int = 4, num_layers: int = 1, heads: int = 8, ratio: int = 4, p: float = 0.1):
        super().__init__()
        self.token = nn.Parameter(torch.randn(num_tokens, dims))
        self.layers = nn.ModuleList([
            BottleneckTransformerLayer(dims, heads=heads, ratio=ratio, p=p)
            for _ in range(num_layers)
        ])
        nn.init.xavier_uniform_(self.token)

    def forward(self, a, b, pe=None, mask=None):
        t = self.token.unsqueeze(0).expand(a.size(0), -1, -1)
        for layer in self.layers:
            a, b, t = layer(a, b, t, pe=pe, mask=mask)
        return a, b


class UniModalEncoder(nn.Module):
    """Encoder for a single modality"""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.mapping = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        x = self.dropout(x)
        x = self.mapping(x)
        pe = self.pos_enc(x)
        x = x + pe
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class CrossModalEncoder(nn.Module):
    """Fuse two modalities"""

    def __init__(self, dims: int, fusion_type: str = 'sum'):
        super().__init__()
        assert fusion_type in ('sum', 'mean', 'concat')
        self.fusion_type = fusion_type
        if fusion_type == 'concat':
            self.mapping = nn.Linear(2 * dims, dims)
        else:
            self.mapping = None
        self.encoder = BottleneckTransformer(dims=dims)
        self.pos_enc = PositionalEncoding(dims)
        self.norm = nn.LayerNorm(dims)

    def forward(self, a, b, mask=None):
        pe = self.pos_enc(a)
        a, b = self.encoder(a, b, pe=pe, mask=mask)
        if self.fusion_type == 'sum':
            x = a + b
        elif self.fusion_type == 'mean':
            x = (a + b) / 2
        else:
            x = torch.cat([a, b], dim=-1)
            x = self.mapping(x)
        x = self.norm(x)
        return x


class UMTFusion(nn.Module):
    """UMT fusion module combining video and audio features."""

    def __init__(self, vid_dim: int, aud_dim: int, hidden_dim: int):
        super().__init__()
        self.video_enc = UniModalEncoder(vid_dim, hidden_dim)
        self.audio_enc = UniModalEncoder(aud_dim, hidden_dim)
        self.cross_enc = CrossModalEncoder(hidden_dim)

    def forward(self, vid, aud, mask=None):
        v = self.video_enc(vid, mask)
        a = self.audio_enc(aud, mask)
        return self.cross_enc(v, a, mask)

