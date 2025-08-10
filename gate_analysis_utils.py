#!/usr/bin/env python3
"""
Gate Analysis Utilities for AVIGATEFusionCustom

이 스크립트는 AVIGATEFusionCustom 모듈의 gate 값을 분석하고 시각화하는 도구들을 제공합니다.
추론 과정에서 각 쿼리마다 attention과 FFN gate 값의 변화를 추적하고 분석할 수 있습니다.

Usage:
    1. 모델에서 gate tracking 활성화
    2. 추론 실행
    3. 이 유틸리티로 결과 분석

Example:
    from gate_analysis_utils import GateAnalyzer
    
    # 모델 로드 및 gate tracking 활성화
    model.avigate_fusion.enable_gate_tracking()
    
    # 추론 실행
    for batch_idx, batch in enumerate(dataloader):
        model.avigate_fusion.set_current_sample_id(f"sample_{batch_idx}")
        outputs = model(batch)
    
    # 분석
    analyzer = GateAnalyzer(model.avigate_fusion)
    analyzer.analyze_all_samples("./gate_analysis_results")
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

class GateAnalyzer:
    """AVIGATEFusionCustom의 gate 값을 분석하는 클래스"""
    
    def __init__(self, avigate_fusion_model):
        """
        Args:
            avigate_fusion_model: AVIGATEFusionCustom 인스턴스
        """
        self.model = avigate_fusion_model
        self.results_dir = None
        
    def analyze_all_samples(self, save_dir: str = "./gate_analysis"):
        """모든 샘플의 gate 값을 종합적으로 분석합니다."""
        if not self.model.track_gates or not self.model.gate_history:
            print("No gate data available. Make sure gate tracking is enabled.")
            return
            
        self.results_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Analyzing gate values for {len(self.model.gate_history)} samples...")
        
        # 1. 개별 샘플 분석
        self._analyze_individual_samples()
        
        # 2. 전체 통계 분석
        self._analyze_overall_statistics()
        
        # 3. 레이어별 gate 분포 분석
        self._analyze_layer_distributions()
        
        # 4. Gate 활성화 패턴 분석
        self._analyze_activation_patterns()
        
        # 5. 상세 gate 분석 (새로 추가)
        self._analyze_detailed_gates()
        
        # 6. 요약 리포트 생성
        self._generate_summary_report()
        
        print(f"Analysis complete. Results saved to {save_dir}")
    
    def _analyze_individual_samples(self):
        """개별 샘플의 gate 값을 분석합니다."""
        print("Analyzing individual samples...")
        
        sample_dir = os.path.join(self.results_dir, "individual_samples")
        os.makedirs(sample_dir, exist_ok=True)
        
        for sample_id in self.model.gate_history.keys():
            self.model._plot_gate_values(sample_id, sample_dir)
            
            # 상세 분석 저장
            self._save_sample_details(sample_id, sample_dir)
    
    def _save_sample_details(self, sample_id: str, save_dir: str):
        """특정 샘플의 상세 분석을 저장합니다."""
        sample_data = self.model.gate_history[sample_id]
        
        details = {
            'sample_id': sample_id,
            'gating_type': self.model.gating_type,
            'num_layers': len(sample_data['attention_gates']),
            'layer_analysis': []
        }
        
        for layer_idx in range(len(sample_data['attention_gates'])):
            att_gates = sample_data['attention_gates'][layer_idx]
            ffn_gates = sample_data['ffn_gates'][layer_idx]
            
            layer_analysis = {
                'layer_index': layer_idx,
                'attention_gate': {
                    'shape': att_gates.shape,
                    'mean': float(np.mean(att_gates)),
                    'std': float(np.std(att_gates)),
                    'min': float(np.min(att_gates)),
                    'max': float(np.max(att_gates)),
                    'median': float(np.median(att_gates)),
                    'percentiles': {
                        '25': float(np.percentile(att_gates, 25)),
                        '75': float(np.percentile(att_gates, 75)),
                        '95': float(np.percentile(att_gates, 95))
                    }
                },
                'ffn_gate': {
                    'shape': ffn_gates.shape,
                    'mean': float(np.mean(ffn_gates)),
                    'std': float(np.std(ffn_gates)),
                    'min': float(np.min(ffn_gates)),
                    'max': float(np.max(ffn_gates)),
                    'median': float(np.median(ffn_gates)),
                    'percentiles': {
                        '25': float(np.percentile(ffn_gates, 25)),
                        '75': float(np.percentile(ffn_gates, 75)),
                        '95': float(np.percentile(ffn_gates, 95))
                    }
                }
            }
            
            details['layer_analysis'].append(layer_analysis)
        
        # JSON으로 저장
        details_file = os.path.join(save_dir, f"{sample_id}_details.json")
        with open(details_file, 'w') as f:
            json.dump(details, f, indent=2)
    
    def _analyze_overall_statistics(self):
        """전체 샘플에 대한 통계를 분석합니다."""
        print("Analyzing overall statistics...")
        
        all_att_gates = []
        all_ffn_gates = []
        layer_stats = {}
        
        for sample_id, sample_data in self.model.gate_history.items():
            for layer_idx in range(len(sample_data['attention_gates'])):
                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = {'att_gates': [], 'ffn_gates': []}
                
                att_gates = sample_data['attention_gates'][layer_idx]
                ffn_gates = sample_data['ffn_gates'][layer_idx]
                
                all_att_gates.extend(att_gates.flatten())
                all_ffn_gates.extend(ffn_gates.flatten())
                
                layer_stats[layer_idx]['att_gates'].extend(att_gates.flatten())
                layer_stats[layer_idx]['ffn_gates'].extend(ffn_gates.flatten())
        
        # 전체 분포 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Attention gate 전체 분포
        axes[0, 0].hist(all_att_gates, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Overall Attention Gate Distribution')
        axes[0, 0].set_xlabel('Gate Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(np.mean(all_att_gates), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_att_gates):.3f}')
        axes[0, 0].legend()
        
        # FFN gate 전체 분포
        axes[0, 1].hist(all_ffn_gates, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Overall FFN Gate Distribution')
        axes[0, 1].set_xlabel('Gate Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(all_ffn_gates), color='red', linestyle='--',
                          label=f'Mean: {np.mean(all_ffn_gates):.3f}')
        axes[0, 1].legend()
        
        # 레이어별 평균 gate 값
        layer_indices = list(layer_stats.keys())
        att_means = [np.mean(layer_stats[i]['att_gates']) for i in layer_indices]
        ffn_means = [np.mean(layer_stats[i]['ffn_gates']) for i in layer_indices]
        
        axes[1, 0].plot(layer_indices, att_means, 'o-', color='blue', label='Attention Gate')
        axes[1, 0].plot(layer_indices, ffn_means, 's-', color='green', label='FFN Gate')
        axes[1, 0].set_title('Mean Gate Values by Layer')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Mean Gate Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 레이어별 gate 값 분산
        att_stds = [np.std(layer_stats[i]['att_gates']) for i in layer_indices]
        ffn_stds = [np.std(layer_stats[i]['ffn_gates']) for i in layer_indices]
        
        axes[1, 1].plot(layer_indices, att_stds, 'o-', color='blue', label='Attention Gate')
        axes[1, 1].plot(layer_indices, ffn_stds, 's-', color='green', label='FFN Gate')
        axes[1, 1].set_title('Gate Value Standard Deviation by Layer')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Standard Deviation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'overall_statistics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 통계 요약 저장
        overall_stats = {
            'total_samples': len(self.model.gate_history),
            'gating_type': self.model.gating_type,
            'num_layers': len(layer_indices),
            'attention_gates': {
                'mean': float(np.mean(all_att_gates)),
                'std': float(np.std(all_att_gates)),
                'min': float(np.min(all_att_gates)),
                'max': float(np.max(all_att_gates)),
                'median': float(np.median(all_att_gates))
            },
            'ffn_gates': {
                'mean': float(np.mean(all_ffn_gates)),
                'std': float(np.std(all_ffn_gates)),
                'min': float(np.min(all_ffn_gates)),
                'max': float(np.max(all_ffn_gates)),
                'median': float(np.median(all_ffn_gates))
            },
            'layer_statistics': {
                str(layer_idx): {
                    'attention_mean': float(np.mean(layer_stats[layer_idx]['att_gates'])),
                    'attention_std': float(np.std(layer_stats[layer_idx]['att_gates'])),
                    'ffn_mean': float(np.mean(layer_stats[layer_idx]['ffn_gates'])),
                    'ffn_std': float(np.std(layer_stats[layer_idx]['ffn_gates']))
                } for layer_idx in layer_indices
            }
        }
        
        with open(os.path.join(self.results_dir, 'overall_statistics.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)
    
    def _analyze_layer_distributions(self):
        """레이어별 gate 분포를 상세히 분석합니다."""
        print("Analyzing layer-wise distributions...")
        
        layer_stats = {}
        for sample_id, sample_data in self.model.gate_history.items():
            for layer_idx in range(len(sample_data['attention_gates'])):
                if layer_idx not in layer_stats:
                    layer_stats[layer_idx] = {'att_gates': [], 'ffn_gates': []}
                
                att_gates = sample_data['attention_gates'][layer_idx]
                ffn_gates = sample_data['ffn_gates'][layer_idx]
                
                layer_stats[layer_idx]['att_gates'].extend(att_gates.flatten())
                layer_stats[layer_idx]['ffn_gates'].extend(ffn_gates.flatten())
        
        num_layers = len(layer_stats)
        fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 10))
        if num_layers == 1:
            axes = axes.reshape(2, 1)
        
        for layer_idx in range(num_layers):
            att_gates = layer_stats[layer_idx]['att_gates']
            ffn_gates = layer_stats[layer_idx]['ffn_gates']
            
            # Attention gate 분포
            axes[0, layer_idx].hist(att_gates, bins=30, alpha=0.7, color='blue', 
                                   edgecolor='black', density=True)
            axes[0, layer_idx].set_title(f'Layer {layer_idx} - Attention Gate Distribution')
            axes[0, layer_idx].set_xlabel('Gate Value')
            axes[0, layer_idx].set_ylabel('Density')
            axes[0, layer_idx].axvline(np.mean(att_gates), color='red', linestyle='--',
                                      label=f'Mean: {np.mean(att_gates):.3f}')
            axes[0, layer_idx].legend()
            
            # FFN gate 분포
            axes[1, layer_idx].hist(ffn_gates, bins=30, alpha=0.7, color='green',
                                   edgecolor='black', density=True)
            axes[1, layer_idx].set_title(f'Layer {layer_idx} - FFN Gate Distribution')
            axes[1, layer_idx].set_xlabel('Gate Value')
            axes[1, layer_idx].set_ylabel('Density')
            axes[1, layer_idx].axvline(np.mean(ffn_gates), color='red', linestyle='--',
                                      label=f'Mean: {np.mean(ffn_gates):.3f}')
            axes[1, layer_idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'layer_distributions.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_activation_patterns(self):
        """Gate 활성화 패턴을 분석합니다."""
        print("Analyzing activation patterns...")
        
        # 샘플별 평균 gate 값 계산
        sample_means = {}
        for sample_id, sample_data in self.model.gate_history.items():
            att_means = []
            ffn_means = []
            
            for layer_idx in range(len(sample_data['attention_gates'])):
                att_mean = np.mean(sample_data['attention_gates'][layer_idx])
                ffn_mean = np.mean(sample_data['ffn_gates'][layer_idx])
                att_means.append(att_mean)
                ffn_means.append(ffn_mean)
            
            sample_means[sample_id] = {
                'attention': att_means,
                'ffn': ffn_means
            }
        
        # 히트맵으로 시각화
        sample_ids = list(sample_means.keys())
        num_layers = len(sample_means[sample_ids[0]]['attention'])
        
        att_matrix = np.array([sample_means[sid]['attention'] for sid in sample_ids])
        ffn_matrix = np.array([sample_means[sid]['ffn'] for sid in sample_ids])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Attention gate 패턴
        im1 = ax1.imshow(att_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax1.set_title('Attention Gate Activation Patterns')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Sample Index')
        ax1.set_xticks(range(num_layers))
        plt.colorbar(im1, ax=ax1, label='Gate Value')
        
        # FFN gate 패턴
        im2 = ax2.imshow(ffn_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_title('FFN Gate Activation Patterns')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Sample Index')
        ax2.set_xticks(range(num_layers))
        plt.colorbar(im2, ax=ax2, label='Gate Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'activation_patterns.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 클러스터링 분석 (옵션)
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            # 모든 gate 값을 하나의 feature 벡터로 결합
            features = np.concatenate([att_matrix, ffn_matrix], axis=1)
            
            # PCA로 차원 축소
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=min(3, len(sample_ids)), random_state=42)
            clusters = kmeans.fit_predict(features)
            
            # 클러스터링 결과 시각화
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=clusters, cmap='tab10')
            plt.title('Gate Activation Pattern Clustering')
            plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
            plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
            plt.colorbar(scatter, label='Cluster')
            
            # 샘플 ID 라벨 추가
            for i, sid in enumerate(sample_ids):
                plt.annotate(sid, (features_2d[i, 0], features_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'gate_clustering.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            print("Scikit-learn not available. Skipping clustering analysis.")
    
    def _analyze_detailed_gates(self):
        """상세한 gate 분석을 수행합니다."""
        print("Analyzing detailed gates...")
        
        # AVIGATEFusionCustom에서 상세 분석 실행
        detailed_analysis = self.model.get_detailed_gate_analysis()
        if detailed_analysis is None:
            print("No detailed gate analysis available.")
            return
        
        # 상세 분석 결과를 JSON으로 저장
        with open(os.path.join(self.results_dir, 'detailed_gate_analysis.json'), 'w') as f:
            json.dump(detailed_analysis, f, indent=2)
        
        # 상세 분석 결과 출력
        self.model.print_detailed_gate_summary()
        
        # 쿼리별 gate 값 분포 시각화
        self._plot_query_gate_distributions(detailed_analysis)
        
        # 레이어별 gate 통계 시각화
        self._plot_layer_gate_statistics(detailed_analysis)
    
    def _plot_query_gate_distributions(self, detailed_analysis):
        """쿼리별 gate 값 분포를 시각화합니다."""
        rankings = detailed_analysis['query_rankings']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MHA Top 10
        mha_top_qids, mha_top_values = zip(*rankings['mha_gates']['top_10'])
        ax1.bar(range(len(mha_top_qids)), mha_top_values, color='skyblue')
        ax1.set_title('Top 10 MHA Gate Values')
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Gate Value')
        ax1.set_xticks(range(len(mha_top_qids)))
        ax1.set_xticklabels([f'Q{qid}' for qid in mha_top_qids], rotation=45)
        
        # MHA Bottom 10
        mha_bot_qids, mha_bot_values = zip(*rankings['mha_gates']['bottom_10'])
        ax2.bar(range(len(mha_bot_qids)), mha_bot_values, color='lightcoral')
        ax2.set_title('Bottom 10 MHA Gate Values')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Gate Value')
        ax2.set_xticks(range(len(mha_bot_qids)))
        ax2.set_xticklabels([f'Q{qid}' for qid in mha_bot_qids], rotation=45)
        
        # FFN Top 10
        ffn_top_qids, ffn_top_values = zip(*rankings['ffn_gates']['top_10'])
        ax3.bar(range(len(ffn_top_qids)), ffn_top_values, color='lightgreen')
        ax3.set_title('Top 10 FFN Gate Values')
        ax3.set_xlabel('Rank')
        ax3.set_ylabel('Gate Value')
        ax3.set_xticks(range(len(ffn_top_qids)))
        ax3.set_xticklabels([f'Q{qid}' for qid in ffn_top_qids], rotation=45)
        
        # FFN Bottom 10
        ffn_bot_qids, ffn_bot_values = zip(*rankings['ffn_gates']['bottom_10'])
        ax4.bar(range(len(ffn_bot_qids)), ffn_bot_values, color='lightsalmon')
        ax4.set_title('Bottom 10 FFN Gate Values')
        ax4.set_xlabel('Rank')
        ax4.set_ylabel('Gate Value')
        ax4.set_xticks(range(len(ffn_bot_qids)))
        ax4.set_xticklabels([f'Q{qid}' for qid in ffn_bot_qids], rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'query_gate_rankings.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_gate_statistics(self, detailed_analysis):
        """레이어별 gate 통계를 시각화합니다."""
        layer_stats = detailed_analysis['layer_gate_statistics']
        
        layers = list(layer_stats.keys())
        mha_means = [layer_stats[layer]['mha_statistics']['mean'] for layer in layers]
        mha_stds = [layer_stats[layer]['mha_statistics']['std'] for layer in layers]
        ffn_means = [layer_stats[layer]['ffn_statistics']['mean'] for layer in layers]
        ffn_stds = [layer_stats[layer]['ffn_statistics']['std'] for layer in layers]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # MHA 평균값
        ax1.bar([f'L{layer}' for layer in layers], mha_means, color='skyblue', alpha=0.7)
        ax1.set_title('MHA Gate Mean Values by Layer')
        ax1.set_ylabel('Mean Gate Value')
        ax1.grid(True, alpha=0.3)
        
        # MHA 표준편차
        ax2.bar([f'L{layer}' for layer in layers], mha_stds, color='navy', alpha=0.7)
        ax2.set_title('MHA Gate Standard Deviation by Layer')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        # FFN 평균값
        ax3.bar([f'L{layer}' for layer in layers], ffn_means, color='lightgreen', alpha=0.7)
        ax3.set_title('FFN Gate Mean Values by Layer')
        ax3.set_ylabel('Mean Gate Value')
        ax3.grid(True, alpha=0.3)
        
        # FFN 표준편차
        ax4.bar([f'L{layer}' for layer in layers], ffn_stds, color='darkgreen', alpha=0.7)
        ax4.set_title('FFN Gate Standard Deviation by Layer')
        ax4.set_ylabel('Standard Deviation')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'layer_gate_statistics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary_report(self):
        """분석 결과 요약 리포트를 생성합니다."""
        print("Generating summary report...")
        
        # 전체 통계 로드
        with open(os.path.join(self.results_dir, 'overall_statistics.json'), 'r') as f:
            stats = json.load(f)
        
        report = f"""
# Gate Analysis Summary Report

## Dataset Information
- **Total Samples**: {stats['total_samples']}
- **Gating Type**: {stats['gating_type']}
- **Number of Layers**: {stats['num_layers']}

## Overall Gate Statistics

### Attention Gates
- **Mean**: {stats['attention_gates']['mean']:.4f}
- **Standard Deviation**: {stats['attention_gates']['std']:.4f}
- **Range**: [{stats['attention_gates']['min']:.4f}, {stats['attention_gates']['max']:.4f}]
- **Median**: {stats['attention_gates']['median']:.4f}

### FFN Gates
- **Mean**: {stats['ffn_gates']['mean']:.4f}
- **Standard Deviation**: {stats['ffn_gates']['std']:.4f}
- **Range**: [{stats['ffn_gates']['min']:.4f}, {stats['ffn_gates']['max']:.4f}]
- **Median**: {stats['ffn_gates']['median']:.4f}

## Layer-wise Analysis

"""
        
        for layer_key, layer_stats in stats['layer_statistics'].items():
            # layer_key는 문자열이므로 적절히 변환
            layer_idx = layer_key if isinstance(layer_key, str) else str(layer_key)
            report += f"""
### Layer {layer_idx}
- **Attention Gate**: Mean = {layer_stats['attention_mean']:.4f}, Std = {layer_stats['attention_std']:.4f}
- **FFN Gate**: Mean = {layer_stats['ffn_mean']:.4f}, Std = {layer_stats['ffn_std']:.4f}
"""
        
        report += f"""

## Key Insights

1. **Gate Activation Level**: 
   - Attention gates have an average activation of {stats['attention_gates']['mean']:.3f}
   - FFN gates have an average activation of {stats['ffn_gates']['mean']:.3f}

2. **Variability**: 
   - Attention gate variability (std): {stats['attention_gates']['std']:.3f}
   - FFN gate variability (std): {stats['ffn_gates']['std']:.3f}

3. **Layer Progression**: 
   - Gate values show layer-wise progression patterns across the network

## Files Generated
- `overall_statistics.png`: Overall distribution and layer-wise trends
- `layer_distributions.png`: Detailed layer-wise gate distributions  
- `activation_patterns.png`: Heatmap of gate activation patterns across samples
- `gate_clustering.png`: Clustering analysis of gate patterns (if available)
- `individual_samples/`: Individual sample analyses
- `overall_statistics.json`: Detailed statistics in JSON format

## Recommendations

Based on the analysis:
- Gate values are {'well-distributed' if stats['attention_gates']['std'] > 0.1 else 'concentrated'} around the mean
- {'Audio information is being effectively gated' if stats['attention_gates']['mean'] > 0.3 else 'Low gate activation suggests limited audio contribution'}
- {'FFN gating is active' if stats['ffn_gates']['mean'] > 0.3 else 'FFN gating shows limited activation'}
"""
        
        # 리포트 저장
        with open(os.path.join(self.results_dir, 'ANALYSIS_REPORT.md'), 'w') as f:
            f.write(report)
        
        print("Summary report saved as ANALYSIS_REPORT.md")

def run_gate_analysis_example():
    """Gate 분석 사용 예시를 실행합니다."""
    print("""
=== Gate Analysis Usage Example ===

To use this gate analysis system:

1. Enable gate tracking in your model:
   ```python
   model.avigate_fusion.enable_gate_tracking()
   ```

2. Set sample IDs during inference:
   ```python
   for batch_idx, batch in enumerate(dataloader):
       model.avigate_fusion.set_current_sample_id(f"sample_{batch_idx}")
       outputs = model(batch)
   ```

3. Run analysis:
   ```python
   from gate_analysis_utils import GateAnalyzer
   analyzer = GateAnalyzer(model.avigate_fusion)
   analyzer.analyze_all_samples("./gate_analysis_results")
   ```

4. Check results in the output directory:
   - Individual sample plots
   - Overall statistics
   - Distribution analysis
   - Activation patterns
   - Summary report

For quick gate value checking during development:
   ```python
   model.avigate_fusion.print_gate_summary()
   ```
""")

if __name__ == "__main__":
    run_gate_analysis_example()
