#!/usr/bin/env python3
"""
Quick Gate Analysis Test

AVIGATEFusionCustom의 gate 분석 기능을 빠르게 테스트할 수 있는 스크립트입니다.
실제 모델이나 데이터 없이도 gate 분석 시스템의 동작을 확인할 수 있습니다.

Usage:
    python test_gate_analysis.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qd_detr.avigate_custom import AVIGATEFusionCustom
from gate_analysis_utils import GateAnalyzer

def create_test_model(gating_type='global', num_layers=2):
    """테스트용 모델을 생성합니다."""
    model = AVIGATEFusionCustom(
        vid_dim=256,
        aud_dim=256,
        hidden_dim=256,
        n_heads=8,
        num_layers=num_layers,
        gating_type=gating_type
    )
    model.eval()
    return model

def generate_test_data(batch_size=2, seq_len=75, hidden_dim=256, num_samples=5):
    """테스트용 데이터를 생성합니다."""
    test_data = []
    
    for i in range(num_samples):
        video_feat = torch.randn(batch_size, seq_len, hidden_dim)
        audio_feat = torch.randn(batch_size, seq_len, hidden_dim)
        
        # 다양한 패턴의 데이터 생성
        if i % 3 == 0:
            # 강한 audio 신호
            audio_feat = audio_feat * 2.0
        elif i % 3 == 1:
            # 약한 audio 신호
            audio_feat = audio_feat * 0.5
        else:
            # 일반적인 신호
            pass
        
        test_data.append({
            'video_feat': video_feat,
            'audio_feat': audio_feat,
            'sample_id': f'test_sample_{i}'
        })
    
    return test_data

def run_test_inference(model, test_data):
    """테스트 추론을 실행합니다."""
    print("Running test inference...")
    
    # Gate 추적 활성화
    model.enable_gate_tracking()
    
    results = []
    with torch.no_grad():
        for data in test_data:
            # 샘플 ID 설정
            model.set_current_sample_id(data['sample_id'])
            
            # 추론 실행
            fused_feat = model(data['video_feat'], data['audio_feat'])
            
            results.append({
                'sample_id': data['sample_id'],
                'output_shape': fused_feat.shape,
                'fused_feat': fused_feat
            })
    
    return results

def test_all_gating_types():
    """모든 gating type을 테스트합니다."""
    gating_types = ['global', 'clipwise', 'elementwise']
    
    for gating_type in gating_types:
        print(f"\n{'='*20} Testing {gating_type.upper()} Gating {'='*20}")
        
        # 모델 생성
        model = create_test_model(gating_type=gating_type, num_layers=2)
        
        # 테스트 데이터 생성
        test_data = generate_test_data(batch_size=1, num_samples=3)
        
        # 추론 실행
        results = run_test_inference(model, test_data)
        
        # 간단한 통계 출력
        model.print_gate_summary()
        
        # 분석 실행
        output_dir = f"./test_results_{gating_type}"
        analyzer = GateAnalyzer(model)
        analyzer.analyze_all_samples(output_dir)
        
        print(f"Results saved to {output_dir}")
        
        # 메모리 정리
        del model, test_data, results

def test_gate_comparison():
    """여러 샘플의 gate 값 비교 테스트"""
    print(f"\n{'='*20} Testing Gate Comparison {'='*20}")
    
    model = create_test_model(gating_type='global', num_layers=3)
    test_data = generate_test_data(batch_size=1, num_samples=10)
    
    # 추론 실행
    results = run_test_inference(model, test_data)
    
    # 특정 샘플들 비교
    sample_ids = [data['sample_id'] for data in test_data[:5]]
    comparison_data = model.compare_samples_gates(sample_ids, "./test_comparison")
    
    print("Gate comparison completed!")
    print(f"Comparison data keys: {list(comparison_data.keys())}")
    
    return comparison_data

def test_detailed_analysis():
    """상세 분석 기능 테스트"""
    print(f"\n{'='*20} Testing Detailed Analysis {'='*20}")
    
    model = create_test_model(gating_type='clipwise', num_layers=2)
    
    # 더 큰 배치로 테스트
    test_data = generate_test_data(batch_size=3, seq_len=100, num_samples=7)
    
    # 추론 실행
    results = run_test_inference(model, test_data)
    
    # 상세 분석
    analyzer = GateAnalyzer(model)
    analyzer.analyze_all_samples("./test_detailed_analysis")
    
    # 통계 정보 출력
    stats = model.get_gate_statistics()
    print(f"Statistics for {len(stats)} samples:")
    
    for sample_id, sample_stats in list(stats.items())[:2]:  # 처음 2개 샘플만 출력
        print(f"\nSample {sample_id}:")
        for layer_idx, (att_stat, ffn_stat) in enumerate(zip(
            sample_stats['attention_stats'], sample_stats['ffn_stats'])):
            print(f"  Layer {layer_idx}:")
            print(f"    Attention: mean={att_stat['mean']:.4f}, std={att_stat['std']:.4f}")
            print(f"    FFN:       mean={ffn_stat['mean']:.4f}, std={ffn_stat['std']:.4f}")

def test_gate_evolution():
    """Gate 값의 진화 패턴 테스트"""
    print(f"\n{'='*20} Testing Gate Evolution {'='*20}")
    
    # 점진적으로 변화하는 오디오 신호로 테스트
    model = create_test_model(gating_type='global', num_layers=1)
    model.enable_gate_tracking()
    
    gate_evolution = []
    
    with torch.no_grad():
        for strength in np.linspace(0.1, 2.0, 10):
            video_feat = torch.randn(1, 50, 256)
            audio_feat = torch.randn(1, 50, 256) * strength
            
            sample_id = f"strength_{strength:.2f}"
            model.set_current_sample_id(sample_id)
            
            fused_feat = model(video_feat, audio_feat)
            
            # 현재 gate 값 추출
            sample_data = model.gate_history[sample_id]
            att_gate = np.mean(sample_data['attention_gates'][0])
            ffn_gate = np.mean(sample_data['ffn_gates'][0])
            
            gate_evolution.append({
                'strength': strength,
                'att_gate': att_gate,
                'ffn_gate': ffn_gate
            })
    
    # 진화 패턴 시각화
    strengths = [item['strength'] for item in gate_evolution]
    att_gates = [item['att_gate'] for item in gate_evolution]
    ffn_gates = [item['ffn_gate'] for item in gate_evolution]
    
    plt.figure(figsize=(10, 6))
    plt.plot(strengths, att_gates, 'o-', label='Attention Gate', color='blue')
    plt.plot(strengths, ffn_gates, 's-', label='FFN Gate', color='green')
    plt.xlabel('Audio Signal Strength')
    plt.ylabel('Gate Value')
    plt.title('Gate Value Evolution with Audio Signal Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./test_gate_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Gate evolution analysis completed!")
    print(f"Attention gate range: [{min(att_gates):.3f}, {max(att_gates):.3f}]")
    print(f"FFN gate range: [{min(ffn_gates):.3f}, {max(ffn_gates):.3f}]")

def run_comprehensive_test():
    """종합 테스트 실행"""
    print("Starting comprehensive gate analysis test...")
    print("This will test all major features of the gate analysis system.")
    
    # 1. 모든 gating type 테스트
    test_all_gating_types()
    
    # 2. Gate 비교 테스트
    test_gate_comparison()
    
    # 3. 상세 분석 테스트
    test_detailed_analysis()
    
    # 4. Gate 진화 패턴 테스트
    test_gate_evolution()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("- test_results_*/ : Analysis results for different gating types")
    print("- test_comparison/ : Sample comparison results")
    print("- test_detailed_analysis/ : Detailed analysis results")
    print("- test_gate_evolution.png : Gate evolution plot")
    print("\nYou can now examine these files to understand how the gate")
    print("analysis system works and apply it to your actual model.")

def quick_demo():
    """빠른 데모 실행"""
    print("Running quick demo of gate analysis...")
    
    # 간단한 모델과 데이터
    model = create_test_model(gating_type='global', num_layers=1)
    test_data = generate_test_data(batch_size=1, num_samples=3)
    
    # 추론 실행
    results = run_test_inference(model, test_data)
    
    # 기본 분석
    model.print_gate_summary()
    
    # 간단한 시각화
    analyzer = GateAnalyzer(model)
    analyzer.analyze_all_samples("./quick_demo_results")
    
    print("Quick demo completed! Check ./quick_demo_results/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gate Analysis System")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick demo only")
    parser.add_argument("--comprehensive", action="store_true",
                       help="Run comprehensive test suite")
    
    args = parser.parse_args()
    
    if args.quick:
        quick_demo()
    elif args.comprehensive:
        run_comprehensive_test()
    else:
        print("Choose an option:")
        print("  --quick: Run a quick demo")
        print("  --comprehensive: Run full test suite")
        print("\nOr run specific tests:")
        
        choice = input("Enter choice (quick/comprehensive/custom): ").strip().lower()
        
        if choice == "quick":
            quick_demo()
        elif choice == "comprehensive":
            run_comprehensive_test()
        elif choice == "custom":
            print("Available tests:")
            print("1. All gating types")
            print("2. Gate comparison")
            print("3. Detailed analysis")
            print("4. Gate evolution")
            
            test_choice = input("Enter test number (1-4): ").strip()
            
            if test_choice == "1":
                test_all_gating_types()
            elif test_choice == "2":
                test_gate_comparison()
            elif test_choice == "3":
                test_detailed_analysis()
            elif test_choice == "4":
                test_gate_evolution()
            else:
                print("Invalid choice")
        else:
            quick_demo()  # 기본값
