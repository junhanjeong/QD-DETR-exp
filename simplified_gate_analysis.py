#!/usr/bin/env python3
"""
Simplified Gat        #         # 4. 결과 저장
        self.results = {
            'layer_gate_averages': layer_gate_averages,
            'sample_rankings': sample_rankings,
            'raw_gat        print("Results saved to {self.output_dir}")
    
    def _create_visualizations(self):
        """시각화 생성"""
        plt.style.use('default')
        
        # 1. Layer별 Gate Type별 평균값 막대 그래프
        self._plot_layer_averages()
        
        # 2. 샘플 랭킹 시각화
        self._plot_sample_rankings()
        
        # 3. 전체 분포 히트맵
        self._plot_gate_heatmap()ata,
            'metadata': {
                'total_samples': len(gate_history),
                'total_layers': len(layer_gate_averages) // 2,
                'ranking_count': sample_rankings['ranking_count'],
                'analysis_timestamp': datetime.now().isoformat()
            }
        }Gate Type별 평균값 계산 (8개 스칼라 값)
        layer_gate_averages = self._calculate_layer_gate_averages(all_data)
        
        # 3. 샘플별 전체 레이어 평균값 계산 및 랭킹
        sample_rankings = self._calculate_sample_rankings(all_data)lysis System
사용자 요구사항에 맞춘 간소화된 게이트 분석 시스템
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# 프로젝트 루트를 Python path에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qd_detr.avigate_custom import AVIGATEFusionCustom
from legacy_model_adapter import load_legacy_model_with_gate_tracking


class SimplifiedGateAnalyzer:
    """간소화된 게이트 분석기"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.results = {}
        
    def analyze_gates(self, model):
        """게이트 값들을 분석하여 요구사항에 맞는 결과를 생성"""
        if not hasattr(model, 'avigate_fusion') or not model.avigate_fusion.gate_history:
            print("No gate data found!")
            return
            
        gate_history = model.avigate_fusion.gate_history
        
        # 1. 모든 게이트 값 수집
        all_gate_data = self._collect_all_gate_data(gate_history)
        
        # 2. Layer별, Gate Type별 평균값 계산 (8개 스칼라 값)
        layer_gate_averages = self._calculate_layer_gate_averages(all_gate_data)
        
        # 3. QID별 전체 레이어 평균값 계산 및 랭킹
        qid_rankings = self._calculate_qid_rankings(all_gate_data)
        
        # 4. 결과 저장
        self.results = {
            'layer_gate_averages': layer_gate_averages,
            'qid_rankings': qid_rankings,
            'raw_gate_data': all_gate_data,
            'metadata': {
                'total_samples': len(gate_history),
                'total_layers': len(layer_gate_averages) // 2,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        # 5. 파일 저장 및 시각화
        self._save_results()
        self._create_visualizations()
        
    def _collect_all_gate_data(self, gate_history):
        """모든 게이트 데이터를 수집"""
        all_data = {
            'samples': {},
            'layer_count': 0,
            'max_seq_len': 0
        }
        
        for sample_id, sample_data in gate_history.items():
            sample_gates = {
                'mha_gates': [],  # [layer][query] = scalar
                'ffn_gates': []   # [layer][query] = scalar
            }
            
            # 각 레이어에서 게이트 값 추출
            for layer_idx in range(len(sample_data['attention_gates'])):
                mha_gate = sample_data['attention_gates'][layer_idx]  # [batch, seq, hidden]
                ffn_gate = sample_data['ffn_gates'][layer_idx]        # [batch, seq, hidden]
                
                # 각 쿼리별로 평균값 계산 (hidden_dim에 대해 평균)
                mha_query_averages = []
                ffn_query_averages = []
                
                seq_len = mha_gate.shape[1]
                for query_idx in range(seq_len):
                    mha_avg = float(mha_gate[0, query_idx, :].mean())
                    ffn_avg = float(ffn_gate[0, query_idx, :].mean())
                    mha_query_averages.append(mha_avg)
                    ffn_query_averages.append(ffn_avg)
                
                sample_gates['mha_gates'].append(mha_query_averages)
                sample_gates['ffn_gates'].append(ffn_query_averages)
                
                all_data['max_seq_len'] = max(all_data['max_seq_len'], seq_len)
            
            all_data['samples'][sample_id] = sample_gates
            all_data['layer_count'] = max(all_data['layer_count'], len(sample_gates['mha_gates']))
        
        return all_data
    
    def _calculate_layer_gate_averages(self, all_data):
        """Layer별, Gate Type별 평균값 계산 (8개 스칼라 값)"""
        layer_count = all_data['layer_count']
        layer_averages = {}
        
        for layer_idx in range(layer_count):
            # 모든 샘플, 모든 쿼리에서 해당 레이어의 게이트 값들 수집
            mha_values = []
            ffn_values = []
            
            for sample_id, sample_data in all_data['samples'].items():
                if layer_idx < len(sample_data['mha_gates']):
                    mha_values.extend(sample_data['mha_gates'][layer_idx])
                    ffn_values.extend(sample_data['ffn_gates'][layer_idx])
            
            # 평균값 계산
            layer_averages[f'layer_{layer_idx}_mha'] = float(np.mean(mha_values)) if mha_values else 0.0
            layer_averages[f'layer_{layer_idx}_ffn'] = float(np.mean(ffn_values)) if ffn_values else 0.0
        
        return layer_averages
    
    def _calculate_sample_rankings(self, all_data):
        """샘플별 전체 레이어 평균값 계산 및 랭킹"""
        layer_count = all_data['layer_count']
        
        sample_averages = {
            'mha': {},  # sample_id -> average value across all layers
            'ffn': {}   # sample_id -> average value across all layers
        }
        
        # 각 샘플별로 모든 레이어의 평균 계산
        for sample_id, sample_data in all_data['samples'].items():
            mha_values = []
            ffn_values = []
            
            # Global gating에서는 모든 clip이 동일한 값이므로 첫 번째 값만 사용
            for layer_idx in range(layer_count):
                if layer_idx < len(sample_data['mha_gates']):
                    # Global gating: 모든 clip이 동일하므로 첫 번째 값만 취함
                    mha_values.append(sample_data['mha_gates'][layer_idx][0])
                    ffn_values.append(sample_data['ffn_gates'][layer_idx][0])
            
            if mha_values:
                sample_averages['mha'][sample_id] = float(np.mean(mha_values))
            if ffn_values:
                sample_averages['ffn'][sample_id] = float(np.mean(ffn_values))
        
        # 랭킹 계산 (샘플 수에 따라 조정)
        rankings = {}
        total_samples = len(sample_averages['mha'])
        top_k = min(10, total_samples)  # 샘플 수가 10개 미만이면 전체 사용
        
        for gate_type in ['mha', 'ffn']:
            sorted_samples = sorted(sample_averages[gate_type].items(), key=lambda x: x[1])
            
            rankings[f'{gate_type}_bottom_{top_k}'] = [
                {'sample_id': sample_id, 'value': value} 
                for sample_id, value in sorted_samples[:top_k]
            ]
            rankings[f'{gate_type}_top_{top_k}'] = [
                {'sample_id': sample_id, 'value': value} 
                for sample_id, value in sorted_samples[-top_k:][::-1]  # 내림차순
            ]
        
        return {
            'sample_averages': sample_averages,
            'rankings': rankings,
            'total_samples': total_samples,
            'ranking_count': top_k
        }
    
    def _save_results(self):
        """결과를 파일로 저장"""
        # JSON 형태로 전체 결과 저장
        with open(os.path.join(self.output_dir, 'gate_analysis_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # CSV 형태로 layer averages 저장
        layer_df = pd.DataFrame([self.results['layer_gate_averages']]).T
        layer_df.columns = ['Average_Value']
        layer_df.index.name = 'Layer_Gate_Type'
        layer_df.to_csv(os.path.join(self.output_dir, 'layer_gate_averages.csv'))
        
        # QID 랭킹을 CSV로 저장
        rankings = self.results['sample_rankings']['rankings']
        ranking_data = []
        
        for gate_type in ['mha', 'ffn']:
            ranking_count = self.results['metadata']['ranking_count']
            for rank_type in [f'top_{ranking_count}', f'bottom_{ranking_count}']:
                for item in rankings[f'{gate_type}_{rank_type}']:
                    ranking_data.append({
                        'gate_type': gate_type,
                        'rank_type': rank_type,
                        'sample_id': item['sample_id'],
                        'value': item['value']
                    })
        
        ranking_df = pd.DataFrame(ranking_data)
        ranking_df.to_csv(os.path.join(self.output_dir, 'sample_rankings.csv'), index=False)
        
        print(f"Results saved to {self.output_dir}")
    
    def _create_visualizations(self):
        """시각화 생성"""
        plt.style.use('default')
        
        # 1. Layer별 Gate Type별 평균값 막대 그래프
        self._plot_layer_averages()
        
        # 2. QID 랭킹 시각화
        self._plot_qid_rankings()
        
        # 3. 전체 분포 히트맵
        self._plot_gate_heatmap()
    
    def _plot_layer_averages(self):
        """Layer별 Gate Type별 평균값 시각화"""
        layer_averages = self.results['layer_gate_averages']
        
        # 데이터 정리
        layers = []
        gate_types = []
        values = []
        
        for key, value in layer_averages.items():
            parts = key.split('_')
            layer_num = parts[1]
            gate_type = parts[2]
            layers.append(f'Layer {layer_num}')
            gate_types.append(gate_type.upper())
            values.append(value)
        
        # DataFrame 생성
        df = pd.DataFrame({
            'Layer': layers,
            'Gate_Type': gate_types,
            'Value': values
        })
        
        # 시각화
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # 그룹별 막대 그래프
        layer_names = df['Layer'].unique()
        gate_types_unique = df['Gate_Type'].unique()
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        for i, gate_type in enumerate(gate_types_unique):
            values_for_type = df[df['Gate_Type'] == gate_type]['Value'].values
            bars = ax.bar(x + i * width, values_for_type, width, label=gate_type)
            
            # 값 표시
            for bar, value in zip(bars, values_for_type):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Layers')
        ax.set_ylabel('Average Gate Value')
        ax.set_title('Layer-wise Gate Type Averages')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(layer_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'layer_gate_averages.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_rankings(self):
        """샘플 랭킹 시각화"""
        rankings = self.results['sample_rankings']['rankings']
        ranking_count = self.results['metadata']['ranking_count']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sample Rankings by Gate Type', fontsize=16)
        
        plots = [
            ('mha', f'top_{ranking_count}', f'MHA Top {ranking_count}', axes[0, 0]),
            ('mha', f'bottom_{ranking_count}', f'MHA Bottom {ranking_count}', axes[0, 1]),
            ('ffn', f'top_{ranking_count}', f'FFN Top {ranking_count}', axes[1, 0]),
            ('ffn', f'bottom_{ranking_count}', f'FFN Bottom {ranking_count}', axes[1, 1])
        ]
        
        for gate_type, rank_type, title, ax in plots:
            data = rankings[f'{gate_type}_{rank_type}']
            sample_ids = [item['sample_id'] for item in data]
            values = [item['value'] for item in data]
            
            bars = ax.bar(range(len(sample_ids)), values, color='skyblue' if 'top' in rank_type else 'lightcoral')
            
            # 값 표시
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(title)
            ax.set_xlabel('Rank')
            ax.set_ylabel('Gate Value')
            ax.set_xticks(range(len(sample_ids)))
            ax.set_xticklabels([f'{sid}' for sid in sample_ids], rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sample_rankings.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gate_heatmap(self):
        """게이트 값 분포 히트맵"""
        sample_averages = self.results['sample_rankings']['sample_averages']
        
        # 데이터 준비
        sample_ids = sorted(sample_averages['mha'].keys())
        mha_values = [sample_averages['mha'][sid] for sid in sample_ids]
        ffn_values = [sample_averages['ffn'][sid] for sid in sample_ids]
        
        # 히트맵 데이터 생성
        heatmap_data = np.array([mha_values, ffn_values])
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
        
        # 축 설정
        ax.set_xticks(range(len(sample_ids)))
        ax.set_xticklabels([f'{sid}' for sid in sample_ids], rotation=90)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['MHA', 'FFN'])
        ax.set_title('Sample-wise Gate Values Heatmap')
        
        # 컬러바 추가
        plt.colorbar(im, ax=ax, label='Gate Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gate_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("SIMPLIFIED GATE ANALYSIS SUMMARY")
        print("="*60)
        
        # 1. Layer별 Gate Type별 평균 (8개 스칼라 값)
        print("\n1. Layer-wise Gate Type Averages (8 scalar values):")
        print("-" * 50)
        layer_averages = self.results['layer_gate_averages']
        for key, value in sorted(layer_averages.items()):
            print(f"  {key}: {value:.6f}")
        
        # 2. 샘플 랭킹
        print("\n2. Sample Rankings:")
        print("-" * 50)
        rankings = self.results['sample_rankings']['rankings']
        ranking_count = self.results['metadata']['ranking_count']
        total_samples = self.results['metadata']['total_samples']
        
        print(f"Total samples: {total_samples}, Ranking count: {ranking_count}")
        
        for gate_type in ['mha', 'ffn']:
            print(f"\n  {gate_type.upper()} Gate:")
            
            print(f"    Top {ranking_count} Samples:")
            for i, item in enumerate(rankings[f'{gate_type}_top_{ranking_count}'], 1):
                print(f"      {i:2d}. Sample {item['sample_id']}: {item['value']:8.6f}")
            
            print(f"    Bottom {ranking_count} Samples:")
            for i, item in enumerate(rankings[f'{gate_type}_bottom_{ranking_count}'], 1):
                print(f"      {i:2d}. Sample {item['sample_id']}: {item['value']:8.6f}")
        
        print("\n" + "="*60)
        print(f"Files saved to: {self.output_dir}")
        print("  - gate_analysis_results.json (전체 결과)")
        print("  - layer_gate_averages.csv (레이어별 평균)")
        print("  - sample_rankings.csv (샘플 랭킹)")
        print("  - layer_gate_averages.png (레이어별 시각화)")
        print("  - sample_rankings.png (랭킹 시각화)")
        print("  - gate_heatmap.png (히트맵)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Simplified Gate Analysis System')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--legacy_model', action='store_true', help='Use legacy model adapter')
    parser.add_argument('--gating_type', type=str, default='global', choices=['global', 'local'], help='Gating type')
    parser.add_argument('--max_samples', type=int, default=3, help='Maximum number of samples to analyze')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"simplified_gate_analysis_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print("Loading model and data...")
        
        # 레거시 모델 어댑터 사용
        if args.legacy_model:
            model = load_legacy_model_with_gate_tracking(args.model_path, args.gating_type)
            config = {}  # 간단한 분석에서는 config 불필요
        else:
            print("Non-legacy model loading not implemented")
            return 1
        
        # 데이터 로드 및 추론
        print(f"Processing {args.max_samples} samples...")
        
        import torch
        from torch.utils.data import DataLoader
        
        # 간단한 데이터 로더 (기존 코드에서 가져옴)
        samples = []
        with open(args.data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= args.max_samples:
                    break
                data = json.loads(line.strip())
                samples.append(data)
        
        # 모델 추론 (gate tracking 활성화)
        model.eval()
        device = next(model.parameters()).device
        
        # Gate tracking 활성화
        model.avigate_fusion.enable_gate_tracking()
        
        with torch.no_grad():
            for i, sample in enumerate(samples):
                qid = sample['qid']
                
                # 현재 샘플 ID 설정
                model.avigate_fusion.set_current_sample_id(qid)
                
                # 더미 입력 생성 (실제 구현에서는 적절한 feature 로드)
                batch_size = 1
                seq_len = 75
                feat_dim = 256
                
                video_feat = torch.randn(batch_size, seq_len, feat_dim).to(device)
                audio_feat = torch.randn(batch_size, seq_len, feat_dim).to(device)
                video_mask = torch.ones(batch_size, seq_len).bool().to(device)
                audio_mask = torch.ones(batch_size, seq_len).bool().to(device)
                
                # Forward pass
                _ = model.avigate_fusion(video_feat, audio_feat, video_mask, audio_mask)
                
                print(f"Processed sample {i+1}/{len(samples)}: {qid}")
        
        # 분석 수행
        print("Analyzing gates...")
        analyzer = SimplifiedGateAnalyzer(args.output_dir)
        analyzer.analyze_gates(model)
        analyzer.print_summary()
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
