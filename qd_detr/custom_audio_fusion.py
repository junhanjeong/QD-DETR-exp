import torch
import torch.nn as nn


class CustomAudioFusion(nn.Module):
    def __init__(self, hidden_dim, n_heads=8, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # 오디오 프로젝션 및 드롭아웃
        self.audio_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Cross-Attention (Audio -> Text)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.dummy_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # Gate 계산을 위한 Self-Attention
        self.gate_self_attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, audio_feat, text_feat, audio_mask, text_mask):
        # 1. 오디오 프로젝션 및 드롭아웃
        proj_audio = self.dropout(self.audio_proj(audio_feat))

        # 2. Cross-Attention (Audio-Query, Text-Key/Value)
        # 더미 토큰 추가
        batch_size = text_feat.size(0)
        dummy_key = self.dummy_token.expand(batch_size, -1, -1)
        key_with_dummy = torch.cat([text_feat, dummy_key], dim=1)
        zero_dummy = torch.zeros_like(dummy_key)
        value_with_dummy = torch.cat([text_feat, zero_dummy], dim=1)

        # 마스크 확장
        dummy_mask = torch.ones(batch_size, 1, device=text_mask.device, dtype=torch.bool)
        mask_with_dummy = torch.cat([text_mask, dummy_mask], dim=1)

        # key_padding_mask는 True인 위치를 무시하므로, 유효한 토큰(0)을 False로, 패딩/더미(1)를 True로 변환
        cross_attn_mask = (1. - mask_with_dummy).bool()

        attn_output, _ = self.cross_attention(
            query=proj_audio,
            key=key_with_dummy,
            value=value_with_dummy,
            key_padding_mask=cross_attn_mask
        )
        audio_feat_after_attn = self.norm1(proj_audio + attn_output)

        # 3. FFN
        audio_feature_initial = self.norm2(audio_feat_after_attn + self.ffn(audio_feat_after_attn))

        # 4. Gate 값 계산
        self_attn_output, _ = self.gate_self_attention(
            query=audio_feature_initial,
            key=audio_feature_initial,
            value=audio_feature_initial,
            key_padding_mask=(1. - audio_mask).bool()
        )
        pooled_output = self_attn_output.mean(dim=1)
        gate = self.gate_mlp(pooled_output).unsqueeze(-1)

        # 5. Gate 적용
        refined_audio_feature = audio_feature_initial * gate

        return refined_audio_feature