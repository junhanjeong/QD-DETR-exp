import torch
from torch import nn
import torch.nn.functional as F

class GatingFunction(nn.Module):
    def __init__(self, hidden_dim, gating_type='global',
                 mha_temp: float = 1.5,
                 mha_bias_init: float = 1.0,
                 mha_scale_init: float = 1.0,
                 ffn_alpha_init: float = 0.3,
                 ffn_film: bool = False,
                 ffn_beta_init: float = 0.3):
        super().__init__()
        self.gating_type = gating_type
        self.ffn_film = ffn_film

        # MHA gate parameters: sigmoid((raw + bias)/temp) * scale
        self.mha_temp = mha_temp
        self.mha_bias = nn.Parameter(torch.tensor(mha_bias_init, dtype=torch.float32))
        self.mha_scale = nn.Parameter(torch.tensor(mha_scale_init, dtype=torch.float32))

        # FFN gate parameters: tanh(raw) * alpha; optional FiLM shift
        self.ffn_alpha = nn.Parameter(torch.tensor(ffn_alpha_init, dtype=torch.float32))
        if self.ffn_film:
            self.ffn_beta = nn.Parameter(torch.tensor(ffn_beta_init, dtype=torch.float32))

        def make_heads(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim)
            )

        if self.gating_type == 'global':
            self.mlp_mha = make_heads(1)
            self.mlp_ffn = make_heads(1)
            if self.ffn_film:
                self.mlp_ffn_beta = make_heads(1)
        elif self.gating_type == 'clipwise':
            self.mlp_mha = make_heads(1)
            self.mlp_ffn = make_heads(1)
            if self.ffn_film:
                self.mlp_ffn_beta = make_heads(1)
        elif self.gating_type == 'elementwise':
            self.mlp_mha = make_heads(hidden_dim)
            self.mlp_ffn = make_heads(hidden_dim)
            if self.ffn_film:
                self.mlp_ffn_beta = make_heads(hidden_dim)

    def forward(self, video_feat, audio_feat):
        if self.gating_type == 'global':
            # (batch_size, seq_len, dim) -> (batch_size, dim)
            pooled_video = video_feat.mean(dim=1)
            pooled_audio = audio_feat.mean(dim=1)
            joint = torch.cat([pooled_video, pooled_audio], dim=1)

            raw_mha = self.mlp_mha(joint)
            gate_mha = torch.sigmoid((raw_mha + self.mha_bias) / self.mha_temp) * self.mha_scale

            raw_ffn = self.mlp_ffn(joint)
            gate_ffn = torch.tanh(raw_ffn) * self.ffn_alpha
            return gate_mha.unsqueeze(-1), gate_ffn.unsqueeze(-1)  # (bs, 1, 1)

        elif self.gating_type == 'clipwise':
            # (batch, seq_len, dim) → (batch, seq_len, 2*dim)
            joint = torch.cat([video_feat, audio_feat], dim=2)
            raw_mha = self.mlp_mha(joint)
            gate_mha = torch.sigmoid((raw_mha + self.mha_bias) / self.mha_temp)  * self.mha_scale  # (bs, L, 1)
            raw_ffn = self.mlp_ffn(joint)
            gate_ffn = torch.tanh(raw_ffn) * self.ffn_alpha  # (bs, L, 1)
            return gate_mha, gate_ffn

        elif self.gating_type == 'elementwise':
            # (batch_size, seq_len, dim * 2)
            joint = torch.cat([video_feat, audio_feat], dim=2)
            raw_mha = self.mlp_mha(joint)
            gate_mha = torch.sigmoid((raw_mha + self.mha_bias) / self.mha_temp) * self.mha_scale  # (bs, L, D)
            raw_ffn = self.mlp_ffn(joint)
            gate_ffn = torch.tanh(raw_ffn) * self.ffn_alpha  # (bs, L, D)
            return gate_mha, gate_ffn

class GatedFusionBlockCustom(nn.Module):
    def __init__(self, hidden_dim, n_heads, gating_type='global',
                 mha_temp: float = 1.5,
                 mha_bias_init: float = 1.0,
                 mha_scale_init: float = 1.0,
                 ffn_alpha_init: float = 0.3,
                 ffn_film: bool = False,
                 ffn_beta_init: float = 0.3):
        super().__init__()
        self.gating_function = GatingFunction(
            hidden_dim, gating_type,
            mha_temp=mha_temp,
            mha_bias_init=mha_bias_init,
            mha_scale_init=mha_scale_init,
            ffn_alpha_init=ffn_alpha_init,
            ffn_film=ffn_film,
            ffn_beta_init=ffn_beta_init,
        )
        self.ffn_film = ffn_film
        self.gating_type = gating_type
        
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        # δ normalizations for stability
        self.delta_norm_mha = nn.LayerNorm(hidden_dim)
        self.delta_norm_ffn = nn.LayerNorm(hidden_dim)
        
        # Self-Attention for refining
        self.self_attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads


    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        gate_mha, gate_ffn = self.gating_function(video_feat, audio_feat)
        
        bs, seq_len, hidden_dim = video_feat.shape
        
        # Custom Cross-Attention
        # Value from audio features, Query from video features
        # No key, no query-key dot product
        # Attention score is an identity matrix

        norm_audio = self.norm1(audio_feat)
        w = self.a_proj.weight.view(self.n_heads, self.head_dim, hidden_dim)  
        b = self.a_proj.bias.view(self.n_heads, self.head_dim) if self.a_proj.bias is not None else None
        value_heads = [
            F.linear(norm_audio, w[i], b[i] if b is not None else None)
            for i in range(self.n_heads)
        ]  # 리스트 요소 하나당 (bs, seq_len, head_dim)
        value = torch.stack(value_heads, dim=2)            # (bs, seq_len, n_heads, head_dim)
        value = value.permute(0, 2, 1, 3).contiguous()
        # value shape: (bs, n_heads, seq_len, head_dim)
        
        # Since we use identity matrix as attention scores, the output is just the value
        attn_output = value.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, hidden_dim)
        attn_output = self.out_proj(attn_output)

        # 1. Gated Fusion Process with δ normalization and sigmoid MHA gate
        delta_mha = self.delta_norm_mha(attn_output)
        z = video_feat + gate_mha * delta_mha
        
        # FFN with tanh(+α) gating; optional FiLM
        ffn_delta = self.ffn1(self.norm2(z))
        ffn_delta = self.delta_norm_ffn(ffn_delta)
        if self.ffn_film:
            # compute shift (beta) using the same inputs as gating
            if self.gating_type == 'global':
                pooled_video = video_feat.mean(dim=1)
                pooled_audio = audio_feat.mean(dim=1)
                joint = torch.cat([pooled_video, pooled_audio], dim=1)
                raw_beta = self.gating_function.mlp_ffn_beta(joint)
                gate_beta = torch.tanh(raw_beta) * self.gating_function.ffn_beta
                gate_beta = gate_beta.unsqueeze(-1)  # (bs,1,1)
            elif self.gating_type in ('clipwise', 'elementwise'):
                joint = torch.cat([video_feat, audio_feat], dim=2)
                raw_beta = self.gating_function.mlp_ffn_beta(joint)
                gate_beta = torch.tanh(raw_beta) * self.gating_function.ffn_beta
            z_bar = z + ffn_delta * (1 + gate_ffn) + gate_beta
        else:
            z_bar = z + gate_ffn * ffn_delta

        # 2. Refining Process
        # Self-Attention
        refined_z, _ = self.self_attention(
            query=self.norm3(z_bar),
            key=self.norm3(z_bar),
            value=self.norm3(z_bar),
            key_padding_mask=video_mask.eq(0) if video_mask is not None else None
        )
        refined_z = refined_z + z_bar

        # FFN
        final_output = self.ffn2(self.norm4(refined_z)) + refined_z

        return final_output


class AVIGATEFusionCustom(nn.Module):
    def __init__(self, vid_dim, aud_dim, hidden_dim, n_heads=8, num_layers=1, gating_type='global',
                 mha_temp: float = 1.5,
                 mha_bias_init: float = 1.0,
                 mha_scale_init: float = 1.0,
                 ffn_alpha_init: float = 0.3,
                 ffn_film: bool = False,
                 ffn_beta_init: float = 0.3):
        super().__init__()

        self.fusion_layers = nn.ModuleList([
            GatedFusionBlockCustom(
                hidden_dim, n_heads, gating_type=gating_type,
                mha_temp=mha_temp,
                mha_bias_init=mha_bias_init,
                mha_scale_init=mha_scale_init,
                ffn_alpha_init=ffn_alpha_init,
                ffn_film=ffn_film,
                ffn_beta_init=ffn_beta_init,
            ) for _ in range(num_layers)
        ])
    
    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        # Inputs: video_feat and audio_feat already projected in model.py
        fused_feat = video_feat
        for layer in self.fusion_layers:
            fused_feat = layer(fused_feat, audio_feat, video_mask, audio_mask)
            
        return fused_feat
