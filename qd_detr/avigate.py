import torch
from torch import nn
import torch.nn.functional as F

class GatingFunction(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
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

    def forward(self, video_feat, audio_feat):
        # (batch_size, seq_len, dim) -> (batch_size, dim)
        pooled_video = video_feat.mean(dim=1)
        pooled_audio = audio_feat.mean(dim=1)
        
        # (batch_size, dim * 2)
        joint_representation = torch.cat([pooled_video, pooled_audio], dim=1)

        gate_mha = torch.tanh(self.mlp_mha(joint_representation))
        gate_ffn = torch.tanh(self.mlp_ffn(joint_representation))
        
        return gate_mha, gate_ffn

class GatedFusionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.gating_function = GatingFunction(hidden_dim)
        
        # Cross-Attention: Video (Query), Audio (Key, Value)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
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


    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        gate_mha, gate_ffn = self.gating_function(video_feat, audio_feat)

        # 1. Gated Fusion Process
        # Cross-Attention
        attn_output, _ = self.cross_attention(
            query=self.norm1(video_feat),
            key=self.norm1(audio_feat),
            value=self.norm1(audio_feat),
            key_padding_mask=audio_mask
        )
        # Gating
        z = gate_mha.unsqueeze(-1) * attn_output + video_feat
        
        # FFN with Gating
        z_bar = gate_ffn.unsqueeze(-1) * self.ffn1(self.norm2(z)) + z

        # 2. Refining Process
        # Self-Attention
        refined_z, _ = self.self_attention(
            query=self.norm3(z_bar),
            key=self.norm3(z_bar),
            value=self.norm3(z_bar),
            key_padding_mask=video_mask
        )
        refined_z = refined_z + z_bar

        # FFN
        final_output = self.ffn2(self.norm4(refined_z)) + refined_z

        return final_output

class AVIGATEFusion(nn.Module):
    def __init__(self, vid_dim, aud_dim, hidden_dim, n_heads=8, num_layers=1):
        super().__init__()
        self.video_proj = nn.Linear(vid_dim, hidden_dim)
        self.audio_proj = nn.Linear(aud_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.5)

        self.fusion_layers = nn.ModuleList([
            GatedFusionBlock(hidden_dim, n_heads) for _ in range(num_layers)
        ])
    
    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        # Project to hidden_dim and apply dropout
        video_feat = self.dropout(self.video_proj(video_feat))
        audio_feat = self.dropout(self.audio_proj(audio_feat))

        fused_feat = video_feat
        for layer in self.fusion_layers:
            fused_feat = layer(fused_feat, audio_feat, video_mask, audio_mask)
            
        return fused_feat