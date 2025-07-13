import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    """
    Computes the clip-wise gate values based on local similarity,
    query-awareness, and temporal smoothing.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # For 3-1. Local similarity gate
        self.fc_base = nn.Linear(hidden_dim * 2, 1)
        
        # For 3-2. Query-aware refinement
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Shared weight for audio projection in the main fusion
        self.w_a = nn.Linear(hidden_dim, hidden_dim)

        # For 3-4. Hard-soft 혼합 (sharpening)
        self.gamma = nn.Parameter(torch.tensor(10.0))
        self.tau = nn.Parameter(torch.tensor(0.5))

    def forward(self, v_feat, a_feat, text_feat_pooled):
        # 3-1. Local similarity gate (텍스트 무관 "순수 A-V 유사도")
        gate_base = torch.sigmoid(self.fc_base(torch.cat([v_feat, a_feat], dim=-1)))  # (B, 75, 1)

        # 3-2. Query-aware refinement (텍스트와 audio 내용 일치도 측정)
        # B: Batch, L: Sequence Length (75), C: Hidden Dim
        # (B, L, C) -> (B, L, C)
        q_aware_proj = a_feat @ self.w_q.weight.T 
        # (B, L, C)와 (B, C)의 내적 -> (B, L)
        gate_q = torch.sigmoid(torch.einsum('blc,bc->bl', q_aware_proj, text_feat_pooled)).unsqueeze(-1)  # (B, L, 1)

        # 최종 gate (두 단계 모두 통과해야 audio 주입)
        alpha = gate_base * gate_q

        # 3-3. Temporal smoothing (Clip 간 부드러운 변환)
        alpha_transposed = alpha.transpose(1, 2)  # (B, 1, 75)
        alpha_smoothed = F.avg_pool1d(alpha_transposed, kernel_size=3, stride=1, padding=1)
        alpha_smoothed = alpha_smoothed.transpose(1, 2)  # (B, 75, 1)

        # 3-4. Hard-soft 혼합 (sharpening)
        alpha_final = alpha_smoothed * torch.sigmoid(self.gamma * (alpha_smoothed - self.tau))

        return alpha_final

class TemporalGateFusion(nn.Module):
    """
    The main fusion module that integrates video and audio features using the GateNet.
    """
    def __init__(self, vid_dim, aud_dim, hidden_dim):
        super().__init__()
        self.v_proj = nn.Linear(vid_dim, hidden_dim)
        self.a_proj = nn.Linear(aud_dim, hidden_dim)
        self.gate_net = GateNet(hidden_dim)
        # Share the weight with GateNet to be used in the final fusion
        self.w_a = self.gate_net.w_a

    def forward(self, v_feat, a_feat, text_feat_pooled):
        v_feat_proj = self.v_proj(v_feat)
        a_feat_proj = self.a_proj(a_feat)
        
        # Calculate the gate value
        alpha = self.gate_net(v_feat_proj, a_feat_proj, text_feat_pooled)
        
        # Fuse features: V’_t = V_t + α_t·W_a A_t
        fused_feat = v_feat_proj + alpha * self.w_a(a_feat_proj)
        # fused_feat = torch.cat([v_feat_proj, alpha * self.w_a(a_feat_proj)], dim=2)
        
        return fused_feat, alpha