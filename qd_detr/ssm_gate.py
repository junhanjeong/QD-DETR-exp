import torch
import torch.nn as nn
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


from typing import Optional, Tuple


def _apply_mask_time(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """x: (B, T, D), mask: (B, T) with 1=valid, 0=pad. Returns masked x (pads=0)."""
    if mask is None:
        return x
    return x * mask.unsqueeze(-1).to(x.dtype)


def compute_ssm(x: torch.Tensor, mask: Optional[torch.Tensor] = None, diag_subtract: float = 0.1) -> torch.Tensor:
    """Compute cosine self-similarity matrix S = X X^T.
    - x: (B, T, D)
    - mask: (B, T) with 1=valid, 0=pad
    - Returns S in (B, T, T) rescaled to [0, 1] approximately, with diagonal slightly down-weighted.
    """
    x = _apply_mask_time(x, mask)
    x = _l2_normalize(x, dim=-1)
    S = torch.einsum('btd,bsd->bts', x, x)
    if diag_subtract is not None and diag_subtract != 0.0:
        B, T, _ = S.shape
        eye = torch.eye(T, device=S.device, dtype=S.dtype).unsqueeze(0)
        S = S - diag_subtract * eye
    # map from [-1, 1] to [0, 1]
    S = (S + 1.0) * 0.5
    if mask is not None:
        # zero out padded rows/cols
        m = mask.to(S.dtype)
        S = S * m.unsqueeze(2) * m.unsqueeze(1)
    return S


def extract_diagonal_strip(S: torch.Tensor, band_width: int) -> torch.Tensor:
    """Extract diagonal neighborhood strips from SSM.
    - S: (B, T, T)
    - Return: (B, 2w+1, T) where channel k corresponds to offset in [-w, w].
    Fills out-of-bound positions with zeros; aligns left (time 0..T-|k|-1 filled).
    """
    B, T, _ = S.shape
    w = int(band_width)
    out = S.new_zeros((B, 2 * w + 1, T))
    for idx, k in enumerate(range(-w, w + 1)):
        if k >= 0:
            # t in [0, T-k-1]: S[t, t+k]
            if T - k > 0:
                # gather diagonal by shifting columns
                diag_vals = S[:, : T - k, k: T].diagonal(dim1=1, dim2=2)
                out[:, idx, : T - k] = diag_vals
        else:
            m = -k
            if T - m > 0:
                diag_vals = S[:, m: T, : T - m].diagonal(dim1=1, dim2=2)
                out[:, idx, : T - m] = diag_vals
    return out


class DiagonalStripEncoder(nn.Module):
    def __init__(self, in_bands: int, hidden: int = 64, dilations: Tuple[int, ...] = (1, 2, 4), dropout: float = 0.1):
        super().__init__()
        layers = []
        c_in = in_bands
        c = hidden
        # initial channel mixing
        layers.append(nn.Conv1d(c_in, c, kernel_size=1))
        layers.append(nn.ReLU())
        prev_c = c
        for d in dilations:
            pad = d
            layers.append(nn.Conv1d(prev_c, c, kernel_size=3, padding=pad, dilation=d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_c = c
            # light channel mixing between dilated layers
            layers.append(nn.Conv1d(prev_c, c, kernel_size=1))
            layers.append(nn.ReLU())
            prev_c = c
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, time_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, C=in_bands, T); time_mask: (B, T) with 1=valid.
        Returns z: (B, C) with GAP over valid time steps.
        """
        h = self.net(x)  # (B, C, T)
        if time_mask is None:
            z = h.mean(dim=-1)
        else:
            m = time_mask.to(h.dtype).unsqueeze(1)  # (B,1,T)
            z = (h * m).sum(dim=-1) / (m.sum(dim=-1) + 1e-6)
        return z  # (B, C)


class DiagonalStripSSMGate(nn.Module):
    """DiagonalStrip-CNN 기반 글로벌 게이트.
    - 오디오/비디오 각각의 SSM에서 대각 스트립을 추출해 1D dilated CNN으로 요약 후 결합.
    - 출력은 g_mha in [0,1], g_ffn in [-1,1]의 스칼라(배치별) 값.
    """
    def __init__(
        self,
        hidden_dim: int,
        band_width: int = 8,
        enc_channels: int = 64,
        dilations: Tuple[int, ...] = (1, 2, 4),
        diag_subtract: float = 0.1,
        use_video_branch: bool = True,
    ):
        super().__init__()
        self.band_width = int(band_width)
        self.diag_subtract = diag_subtract
        in_bands = 2 * self.band_width + 1

        # 인코더는 가중치 공유(오디오/비디오 동일 구조)로 효율화
        self.encoder = DiagonalStripEncoder(in_bands=in_bands, hidden=enc_channels, dilations=dilations)
        self.use_video_branch = use_video_branch

        feat_in = enc_channels * (2 if use_video_branch else 1)
        self.norm = nn.LayerNorm(feat_in)
        self.mlp_mha = nn.Sequential(
            nn.Linear(feat_in, enc_channels),
            nn.ReLU(),
            nn.Linear(enc_channels, 1),
        )
        self.mlp_ffn = nn.Sequential(
            nn.Linear(feat_in, enc_channels),
            nn.ReLU(),
            nn.Linear(enc_channels, 1),
        )
        # MHA 게이트는 초기 바이어스를 음수로 두어 초반 과주입 방지
        nn.init.constant_(self.mlp_mha[-1].bias, -1.0)

    def forward(
        self,
        video_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # SSM 계산
        S_a = compute_ssm(audio_feat, audio_mask, diag_subtract=self.diag_subtract)  # (B, T_a, T_a)
        strip_a = extract_diagonal_strip(S_a, self.band_width)  # (B, C_bands, T_a)
        z_a = self.encoder(strip_a, time_mask=audio_mask)

        if self.use_video_branch:
            S_v = compute_ssm(video_feat, video_mask, diag_subtract=self.diag_subtract)
            strip_v = extract_diagonal_strip(S_v, self.band_width)
            z_v = self.encoder(strip_v, time_mask=video_mask)
            f = torch.cat([z_a, z_v], dim=-1)
        else:
            f = z_a

        f = self.norm(f)
        g_mha = torch.sigmoid(self.mlp_mha(f))  # (B, 1)
        g_ffn = torch.tanh(self.mlp_ffn(f))     # (B, 1)
        return g_mha.unsqueeze(-1), g_ffn.unsqueeze(-1)  # (B, 1, 1)
