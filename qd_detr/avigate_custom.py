import torch
from torch import nn
import torch.nn.functional as F
from .ssm_gate import DiagonalStripSSMGate

class GatingFunction(nn.Module):
    def __init__(self, hidden_dim, gating_type='global', ssm_band_width=8, ssm_enc_channels=64, ssm_dilations=(1,2,4), ssm_diag_subtract=0.1, ssm_use_video_branch=True):
        super().__init__()
        self.gating_type = gating_type
        self.ssm_cfg = dict(band_width=ssm_band_width,
                            enc_channels=ssm_enc_channels,
                            dilations=tuple(ssm_dilations) if not isinstance(ssm_dilations, tuple) else ssm_dilations,
                            diag_subtract=ssm_diag_subtract,
                            use_video_branch=ssm_use_video_branch)
        
        if self.gating_type == 'global':
            self.mlp_mha = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.mlp_ffn = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            # MHA gate는 sigmoid, FFN gate는 tanh를 사용할 예정
        elif self.gating_type == 'clipwise':
            self.mlp_mha = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # clip별 스칼라
            )
            self.mlp_ffn = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif self.gating_type == 'elementwise':
            self.mlp_mha = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
            self.mlp_ffn = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            )
        elif self.gating_type == 'global_diagstrip_ssm':
            # DiagonalStrip-CNN 기반 SSM 요약을 활용한 글로벌 게이트
            self.ssm_gate = DiagonalStripSSMGate(hidden_dim=hidden_dim, **self.ssm_cfg)

    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        if self.gating_type == 'global':
            # (batch_size, seq_len, dim) -> (batch_size, dim)
            pooled_video = video_feat.mean(dim=1)
            pooled_audio = audio_feat.mean(dim=1)
            
            # (batch_size, dim * 2)
            joint_representation = torch.cat([pooled_video, pooled_audio], dim=1)

            gate_mha = torch.sigmoid(self.mlp_mha(joint_representation))
            gate_ffn = torch.tanh(self.mlp_ffn(joint_representation))
            return gate_mha.unsqueeze(-1), gate_ffn.unsqueeze(-1) # (batch_size, 1, 1) to allow broadcasting

        elif self.gating_type == 'clipwise':
            # (batch, seq_len, dim) → (batch, seq_len, 2*dim)
            joint_representation = torch.cat([video_feat, audio_feat], dim=2)
            gate_mha = torch.tanh(self.mlp_mha(joint_representation))  # (batch, seq_len, 1)
            gate_ffn = torch.tanh(self.mlp_ffn(joint_representation))  # (batch, seq_len, 1)
            return gate_mha, gate_ffn
        
        elif self.gating_type == 'elementwise':
            # (batch_size, seq_len, dim * 2)
            joint_representation = torch.cat([video_feat, audio_feat], dim=2)
            
            gate_mha = torch.tanh(self.mlp_mha(joint_representation)) # (batch_size, seq_len, dim)
            gate_ffn = torch.tanh(self.mlp_ffn(joint_representation)) # (batch_size, seq_len, dim)
            return gate_mha, gate_ffn
        elif self.gating_type == 'global_diagstrip_ssm':
            gate_mha, gate_ffn = self.ssm_gate(video_feat, audio_feat, video_mask=video_mask, audio_mask=audio_mask)
            return gate_mha, gate_ffn

class GatedFusionBlockCustom(nn.Module):
    def __init__(self, hidden_dim, n_heads, gating_type='global', ssm_band_width=8, ssm_enc_channels=64, ssm_dilations=(1,2,4), ssm_diag_subtract=0.1, ssm_use_video_branch=True):
        super().__init__()
        self.gating_function = GatingFunction(hidden_dim, gating_type,
                                             ssm_band_width=ssm_band_width,
                                             ssm_enc_channels=ssm_enc_channels,
                                             ssm_dilations=ssm_dilations,
                                             ssm_diag_subtract=ssm_diag_subtract,
                                             ssm_use_video_branch=ssm_use_video_branch)
        
        self.a_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
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
        gate_mha, gate_ffn = self.gating_function(video_feat, audio_feat, video_mask=video_mask, audio_mask=audio_mask)
        
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

        # 1. Gated Fusion Process
        z = gate_mha * attn_output + video_feat
        
        # FFN with Gating
        z_bar = gate_ffn * self.ffn1(self.norm2(z)) + z

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
                 ssm_band_width=8, ssm_enc_channels=64, ssm_dilations=(1,2,4), ssm_diag_subtract=0.1, ssm_use_video_branch=True):
        super().__init__()

        self.fusion_layers = nn.ModuleList([
            GatedFusionBlockCustom(hidden_dim, n_heads, gating_type=gating_type,
                                   ssm_band_width=ssm_band_width,
                                   ssm_enc_channels=ssm_enc_channels,
                                   ssm_dilations=ssm_dilations,
                                   ssm_diag_subtract=ssm_diag_subtract,
                                   ssm_use_video_branch=ssm_use_video_branch) for _ in range(num_layers)
        ])
    
    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        # Inputs: video_feat and audio_feat already projected in model.py
        fused_feat = video_feat
        for layer in self.fusion_layers:
            fused_feat = layer(fused_feat, audio_feat, video_mask, audio_mask)
            
        return fused_feat
