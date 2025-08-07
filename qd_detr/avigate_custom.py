import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class GatingFunction(nn.Module):
    def __init__(self, hidden_dim, gating_type='global'):
        super().__init__()
        self.gating_type = gating_type
        
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

    def forward(self, video_feat, audio_feat):
        if self.gating_type == 'global':
            # (batch_size, seq_len, dim) -> (batch_size, dim)
            pooled_video = video_feat.mean(dim=1)
            pooled_audio = audio_feat.mean(dim=1)
            
            # (batch_size, dim * 2)
            joint_representation = torch.cat([pooled_video, pooled_audio], dim=1)

            gate_mha = torch.tanh(self.mlp_mha(joint_representation))
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

class GatedFusionBlockCustom(nn.Module):
    def __init__(self, hidden_dim, n_heads, gating_type='global'):
        super().__init__()
        self.gating_function = GatingFunction(hidden_dim, gating_type)
        
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

        return final_output, gate_mha, gate_ffn


class AVIGATEFusionCustom(nn.Module):
    def __init__(self, vid_dim, aud_dim, hidden_dim, n_heads=8, num_layers=1, gating_type='global'):
        super().__init__()

        self.fusion_layers = nn.ModuleList([
            GatedFusionBlockCustom(hidden_dim, n_heads, gating_type=gating_type) for _ in range(num_layers)
        ])
        
        # Gate tracking 관련 속성들
        self.track_gates = False  # gate 값을 추적할지 여부
        self.gate_history = {}  # gate 값들을 저장할 딕셔너리
        self.current_sample_id = None  # 현재 처리 중인 샘플 ID
        self.gating_type = gating_type
        self.num_layers = num_layers
        
    def enable_gate_tracking(self):
        """Gate 값 추적을 활성화합니다."""
        self.track_gates = True
        self.gate_history = {}
        
    def disable_gate_tracking(self):
        """Gate 값 추적을 비활성화합니다."""
        self.track_gates = False
        
    def set_current_sample_id(self, sample_id):
        """현재 처리 중인 샘플의 ID를 설정합니다."""
        self.current_sample_id = sample_id
        if self.track_gates and sample_id not in self.gate_history:
            self.gate_history[sample_id] = {
                'attention_gates': [],  # 각 레이어의 attention gate 값
                'ffn_gates': [],       # 각 레이어의 ffn gate 값
                'layer_info': []       # 각 레이어의 메타데이터
            }
    
    def forward(self, video_feat, audio_feat, video_mask=None, audio_mask=None):
        # Inputs: video_feat and audio_feat already projected in model.py
        fused_feat = video_feat
        
        for layer_idx, layer in enumerate(self.fusion_layers):
            fused_feat, gate_mha, gate_ffn = layer(fused_feat, audio_feat, video_mask, audio_mask)
            
            # Gate 값 추적
            if self.track_gates and self.current_sample_id is not None:
                self._store_gate_values(layer_idx, gate_mha, gate_ffn, video_feat.shape)
            
        return fused_feat
    
    def _store_gate_values(self, layer_idx, gate_mha, gate_ffn, feat_shape):
        """Gate 값들을 저장합니다."""
        bs, seq_len, hidden_dim = feat_shape
        
        # GPU 텐서를 CPU로 이동하고 numpy로 변환
        gate_mha_np = gate_mha.detach().cpu().numpy()
        gate_ffn_np = gate_ffn.detach().cpu().numpy()
        
        sample_history = self.gate_history[self.current_sample_id]
        
        # 레이어별로 gate 값 저장
        if len(sample_history['attention_gates']) <= layer_idx:
            sample_history['attention_gates'].append([])
            sample_history['ffn_gates'].append([])
            sample_history['layer_info'].append({
                'layer_idx': layer_idx,
                'gating_type': self.gating_type,
                'sequence_length': seq_len,
                'batch_size': bs,
                'hidden_dim': hidden_dim
            })
        
        sample_history['attention_gates'][layer_idx] = gate_mha_np
        sample_history['ffn_gates'][layer_idx] = gate_ffn_np
    
    def get_gate_statistics(self, sample_id=None):
        """Gate 값들의 통계를 반환합니다."""
        if not self.track_gates or not self.gate_history:
            return None
            
        if sample_id is None:
            # 모든 샘플의 통계
            sample_ids = list(self.gate_history.keys())
        else:
            sample_ids = [sample_id] if sample_id in self.gate_history else []
            
        if not sample_ids:
            return None
            
        stats = {}
        for sid in sample_ids:
            sample_data = self.gate_history[sid]
            stats[sid] = {
                'attention_stats': [],
                'ffn_stats': [],
                'layer_info': sample_data['layer_info']
            }
            
            for layer_idx in range(len(sample_data['attention_gates'])):
                att_gates = sample_data['attention_gates'][layer_idx]
                ffn_gates = sample_data['ffn_gates'][layer_idx]
                
                att_stats = {
                    'mean': float(np.mean(att_gates)),
                    'std': float(np.std(att_gates)),
                    'min': float(np.min(att_gates)),
                    'max': float(np.max(att_gates)),
                    'shape': att_gates.shape
                }
                
                ffn_stats = {
                    'mean': float(np.mean(ffn_gates)),
                    'std': float(np.std(ffn_gates)),
                    'min': float(np.min(ffn_gates)),
                    'max': float(np.max(ffn_gates)),
                    'shape': ffn_gates.shape
                }
                
                stats[sid]['attention_stats'].append(att_stats)
                stats[sid]['ffn_stats'].append(ffn_stats)
                
        return stats
    
    def save_gate_analysis(self, save_dir, sample_id=None):
        """Gate 값 분석 결과를 저장합니다."""
        if not self.track_gates or not self.gate_history:
            print("Gate tracking is not enabled or no data available.")
            return
            
        os.makedirs(save_dir, exist_ok=True)
        
        if sample_id is None:
            sample_ids = list(self.gate_history.keys())
        else:
            sample_ids = [sample_id] if sample_id in self.gate_history else []
            
        for sid in sample_ids:
            self._plot_gate_values(sid, save_dir)
            
        # 통계 정보를 JSON으로 저장
        stats = self.get_gate_statistics(sample_id)
        if stats:
            import json
            stats_file = os.path.join(save_dir, f"gate_statistics_{sample_id or 'all'}.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
    
    def _plot_gate_values(self, sample_id, save_dir):
        """특정 샘플의 gate 값들을 시각화합니다."""
        if sample_id not in self.gate_history:
            return
            
        sample_data = self.gate_history[sample_id]
        num_layers = len(sample_data['attention_gates'])
        
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        if num_layers == 1:
            axes = axes.reshape(2, 1)
            
        for layer_idx in range(num_layers):
            att_gates = sample_data['attention_gates'][layer_idx]
            ffn_gates = sample_data['ffn_gates'][layer_idx]
            
            # Attention gate 시각화
            ax_att = axes[0, layer_idx]
            if self.gating_type == 'global':
                # Global gating: (batch_size, 1, 1)
                att_values = att_gates.squeeze()
                if att_values.ndim == 0:
                    att_values = [att_values.item()]
                ax_att.bar(range(len(att_values)), att_values)
                ax_att.set_title(f'Layer {layer_idx} - Attention Gate (Global)')
                ax_att.set_ylabel('Gate Value')
                ax_att.set_xlabel('Batch Index')
            else:
                # Clipwise/Elementwise gating: 히트맵으로 표시
                if att_gates.ndim == 3:  # (batch, seq_len, 1) or (batch, seq_len, dim)
                    att_gates_2d = att_gates.squeeze() if att_gates.shape[-1] == 1 else att_gates.mean(axis=-1)
                    im = ax_att.imshow(att_gates_2d.T, aspect='auto', cmap='viridis')
                    ax_att.set_title(f'Layer {layer_idx} - Attention Gate')
                    ax_att.set_ylabel('Sequence Position')
                    ax_att.set_xlabel('Batch Index')
                    plt.colorbar(im, ax=ax_att)
            
            # FFN gate 시각화
            ax_ffn = axes[1, layer_idx]
            if self.gating_type == 'global':
                ffn_values = ffn_gates.squeeze()
                if ffn_values.ndim == 0:
                    ffn_values = [ffn_values.item()]
                ax_ffn.bar(range(len(ffn_values)), ffn_values)
                ax_ffn.set_title(f'Layer {layer_idx} - FFN Gate (Global)')
                ax_ffn.set_ylabel('Gate Value')
                ax_ffn.set_xlabel('Batch Index')
            else:
                if ffn_gates.ndim == 3:
                    ffn_gates_2d = ffn_gates.squeeze() if ffn_gates.shape[-1] == 1 else ffn_gates.mean(axis=-1)
                    im = ax_ffn.imshow(ffn_gates_2d.T, aspect='auto', cmap='viridis')
                    ax_ffn.set_title(f'Layer {layer_idx} - FFN Gate')
                    ax_ffn.set_ylabel('Sequence Position')
                    ax_ffn.set_xlabel('Batch Index')
                    plt.colorbar(im, ax=ax_ffn)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'gate_analysis_{sample_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_gate_summary(self, sample_id=None):
        """Gate 값들의 요약 정보를 출력합니다."""
        stats = self.get_gate_statistics(sample_id)
        if not stats:
            print("No gate statistics available.")
            return
            
        for sid, data in stats.items():
            print(f"\n=== Sample {sid} Gate Summary ===")
            print(f"Gating Type: {self.gating_type}")
            print(f"Number of Layers: {len(data['attention_stats'])}")
            
            for layer_idx, (att_stat, ffn_stat) in enumerate(zip(data['attention_stats'], data['ffn_stats'])):
                print(f"\n--- Layer {layer_idx} ---")
                print(f"Attention Gate - Mean: {att_stat['mean']:.4f}, Std: {att_stat['std']:.4f}, Range: [{att_stat['min']:.4f}, {att_stat['max']:.4f}]")
                print(f"FFN Gate       - Mean: {ffn_stat['mean']:.4f}, Std: {ffn_stat['std']:.4f}, Range: [{ffn_stat['min']:.4f}, {ffn_stat['max']:.4f}]")
                print(f"Shape: Attention {att_stat['shape']}, FFN {ffn_stat['shape']}")
    
    def compare_samples_gates(self, sample_ids, save_dir=None):
        """여러 샘플의 gate 값들을 비교합니다."""
        if not self.track_gates:
            print("Gate tracking is not enabled.")
            return
            
        valid_samples = [sid for sid in sample_ids if sid in self.gate_history]
        if not valid_samples:
            print("No valid sample IDs found.")
            return
            
        # 샘플별 평균 gate 값 계산
        comparison_data = {}
        for sid in valid_samples:
            sample_data = self.gate_history[sid]
            comparison_data[sid] = {
                'att_means': [],
                'ffn_means': []
            }
            
            for layer_idx in range(len(sample_data['attention_gates'])):
                att_mean = np.mean(sample_data['attention_gates'][layer_idx])
                ffn_mean = np.mean(sample_data['ffn_gates'][layer_idx])
                comparison_data[sid]['att_means'].append(att_mean)
                comparison_data[sid]['ffn_means'].append(ffn_mean)
        
        # 비교 시각화
        num_layers = len(comparison_data[valid_samples[0]]['att_means'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Attention gate 비교
        for sid in valid_samples:
            ax1.plot(range(num_layers), comparison_data[sid]['att_means'], 
                    marker='o', label=f'Sample {sid}')
        ax1.set_title('Attention Gate Values Across Layers')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Mean Gate Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # FFN gate 비교
        for sid in valid_samples:
            ax2.plot(range(num_layers), comparison_data[sid]['ffn_means'], 
                    marker='s', label=f'Sample {sid}')
        ax2.set_title('FFN Gate Values Across Layers')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Mean Gate Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'gate_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"Gate comparison plot saved to {save_dir}")
        
        plt.show()
        
        return comparison_data