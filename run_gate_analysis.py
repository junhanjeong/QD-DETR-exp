#!/usr/bin/env python3
"""
Gate Analysis Inference Script

AVIGATEFusionCustom 모듈의 gate 값을 추적하며 추론을 실행하는 스크립트입니다.
이 스크립트는 추론 과정에서 각 쿼리마다 attention과 FFN gate 값을 저장하고
분석할 수 있도록 합니다.

Usage:
    python run_gate_analysis.py --model_path /path/to/model --data_path /path/to/data
"""

import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
import datetime

# 프로젝트 루트 디렉토리를 패스에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from gate_analysis_utils import GateAnalyzer
    from qd_detr.config import BaseOptions
    from qd_detr.model import build_model
    from qd_detr.avigate_custom import AVIGATEFusionCustom
    from legacy_model_adapter import load_legacy_model_with_gate_tracking, adapt_legacy_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def parse_args():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Gate Analysis Inference")
    
    # 모델 관련
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Path to model config file (optional)")
    
    # 데이터 관련
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to evaluation data")
    parser.add_argument("--max_samples", type=int, default=10,
                       help="Maximum number of samples to analyze")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for inference")
    
    # 출력 관련
    parser.add_argument("--output_dir", type=str, default="./gate_analysis_results",
                       help="Directory to save analysis results")
    parser.add_argument("--save_raw_data", action="store_true",
                       help="Save raw gate values as numpy arrays")
    
    # Gate 분석 관련
    parser.add_argument("--analyze_samples", nargs="+", type=str, default=None,
                       help="Specific sample IDs to analyze (space-separated)")
    parser.add_argument("--compare_samples", action="store_true",
                       help="Generate sample comparison plots")
    
    # 모델 설정
    parser.add_argument("--use_avigate_custom", action="store_true",
                       help="Use AVIGATE custom fusion")
    parser.add_argument("--gating_type", type=str, default="global",
                       choices=["global", "clipwise", "elementwise"],
                       help="Gating type for AVIGATE custom")
    parser.add_argument("--legacy_model", action="store_true",
                       help="Use legacy model adapter for old trained models")
    
    return parser.parse_args()

def load_model_and_config(args):
    """모델과 설정을 로드합니다."""
    print("Loading model and configuration...")
    
    # 레거시 모델 처리
    if args.legacy_model:
        print("Loading legacy model with gate tracking support...")
        try:
            model = load_legacy_model_with_gate_tracking(
                args.model_path, 
                gating_type=args.gating_type,
                device='auto'
            )
            config = {
                "use_avigate_custom": True,
                "gating_type": args.gating_type,
                "hidden_dim": 256,
                "nheads": 8,
                "num_queries": 10,
                "legacy_model": True
            }
            device = next(model.parameters()).device
            return model, config, device
            
        except Exception as e:
            print(f"Error loading legacy model: {e}")
            print("Falling back to standard model loading...")
    
    # 기본 설정 로드
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        # 기본 설정 사용
        config = {
            "use_avigate_custom": args.use_avigate_custom,
            "gating_type": args.gating_type,
            "hidden_dim": 256,
            "nheads": 8,
            "num_queries": 10
        }
    
    # 모델 체크포인트 로드
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # 모델 빌드
    try:
        model = build_model(config)
        
        # 체크포인트에서 모델 상태 추출
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
            
        model.load_state_dict(model_state, strict=False)
        
        # Gate 추적 기능이 없는 경우 추가
        if hasattr(model, 'avigate_fusion') and not hasattr(model.avigate_fusion, 'enable_gate_tracking'):
            print("Adapting existing model for gate tracking...")
            model = adapt_legacy_model(model, args.gating_type)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a dummy model for demonstration...")
        model = create_dummy_model(config)
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    return model, config, device

def create_dummy_model(config):
    """테스트용 더미 모델을 생성합니다."""
    class DummyModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.avigate_fusion = AVIGATEFusionCustom(
                vid_dim=config.get('hidden_dim', 256),
                aud_dim=config.get('hidden_dim', 256),
                hidden_dim=config.get('hidden_dim', 256),
                n_heads=config.get('nheads', 8),
                num_layers=config.get('num_layers', 1),
                gating_type=config.get('gating_type', 'global')
            )
            
        def forward(self, batch):
            # 더미 구현
            bs = batch.get('batch_size', 1)
            seq_len = batch.get('seq_len', 75)  # 기본 시퀀스 길이
            hidden_dim = 256
            
            video_feat = torch.randn(bs, seq_len, hidden_dim)
            audio_feat = torch.randn(bs, seq_len, hidden_dim)
            
            fused_feat = self.avigate_fusion(video_feat, audio_feat)
            
            return {'pred_relevant_windows': [[[0, 10, 0.9]] for _ in range(bs)]}
    
    return DummyModel(config)

def create_dummy_dataloader(data_path, max_samples, batch_size):
    """더미 데이터로더를 생성합니다."""
    print(f"Creating dummy dataloader with {max_samples} samples...")
    
    # 실제 데이터가 있다면 로드, 없으면 더미 데이터 생성
    if os.path.exists(data_path):
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} samples from {data_path}")
            data = data[:max_samples]  # 최대 샘플 수 제한
        except:
            print("Failed to load data, using dummy data")
            data = [{'qid': f'dummy_{i}', 'batch_size': batch_size, 'seq_len': 75} 
                   for i in range(max_samples)]
    else:
        print("Data file not found, using dummy data")
        data = [{'qid': f'dummy_{i}', 'batch_size': batch_size, 'seq_len': 75} 
               for i in range(max_samples)]
    
    # 배치로 그룹화
    batches = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch = {
            'batch_size': len(batch_data),
            'seq_len': batch_data[0].get('seq_len', 75),
            'qids': [item.get('qid', f'sample_{i+j}') for j, item in enumerate(batch_data)]
        }
        batches.append(batch)
    
    return batches

def run_inference_with_gate_tracking(model, dataloader, device, output_dir):
    """Gate 추적을 활성화하고 추론을 실행합니다."""
    print("Starting inference with gate tracking...")
    
    # Gate 추적 활성화
    if hasattr(model, 'avigate_fusion') and hasattr(model.avigate_fusion, 'enable_gate_tracking'):
        model.avigate_fusion.enable_gate_tracking()
        print("Gate tracking enabled!")
    else:
        print("Warning: Model does not support gate tracking")
        return None
    
    # 추론 실행
    results = []
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # 배치 내 각 샘플에 대해 개별적으로 처리
            for sample_idx, qid in enumerate(batch['qids']):
                # 샘플 ID 설정
                model.avigate_fusion.set_current_sample_id(qid)
                
                # 단일 샘플 배치 생성
                single_batch = {
                    'batch_size': 1,
                    'seq_len': batch['seq_len'],
                    'qids': [qid]
                }
                
                # 추론 실행
                outputs = model(single_batch)
                
                # 결과 저장
                results.append({
                    'qid': qid,
                    'batch_idx': batch_idx,
                    'sample_idx': sample_idx,
                    'outputs': outputs
                })
    
    print(f"Inference completed. Processed {len(results)} samples.")
    return results

def analyze_gate_values(model, output_dir, args):
    """수집된 gate 값들을 분석합니다."""
    print("Analyzing gate values...")
    
    # GateAnalyzer 초기화
    analyzer = GateAnalyzer(model.avigate_fusion)
    
    # 전체 분석 실행
    analyzer.analyze_all_samples(output_dir)
    
    # 특정 샘플 분석
    if args.analyze_samples:
        print(f"Analyzing specific samples: {args.analyze_samples}")
        for sample_id in args.analyze_samples:
            if sample_id in model.avigate_fusion.gate_history:
                model.avigate_fusion.print_gate_summary(sample_id)
                model.avigate_fusion.save_gate_analysis(
                    os.path.join(output_dir, f"sample_{sample_id}"), sample_id)
    
    # 샘플 비교 분석
    if args.compare_samples and len(model.avigate_fusion.gate_history) > 1:
        sample_ids = list(model.avigate_fusion.gate_history.keys())[:5]  # 최대 5개 샘플 비교
        print(f"Comparing samples: {sample_ids}")
        model.avigate_fusion.compare_samples_gates(sample_ids, 
                                                  os.path.join(output_dir, "comparisons"))
    
    # 원시 데이터 저장
    if args.save_raw_data:
        raw_data_dir = os.path.join(output_dir, "raw_data")
        os.makedirs(raw_data_dir, exist_ok=True)
        
        for sample_id, sample_data in model.avigate_fusion.gate_history.items():
            import numpy as np
            for layer_idx in range(len(sample_data['attention_gates'])):
                np.save(os.path.join(raw_data_dir, f"{sample_id}_layer{layer_idx}_att.npy"),
                       sample_data['attention_gates'][layer_idx])
                np.save(os.path.join(raw_data_dir, f"{sample_id}_layer{layer_idx}_ffn.npy"),
                       sample_data['ffn_gates'][layer_idx])
        
        print(f"Raw gate data saved to {raw_data_dir}")

def main():
    """메인 함수"""
    args = parse_args()
    
    print("="*50)
    print("AVIGATEFusionCustom Gate Analysis")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples: {args.max_samples}")
    print(f"Gating type: {args.gating_type}")
    print(f"Legacy model: {args.legacy_model}")
    print("="*50)
    
    # 출력 디렉토리 생성
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"gate_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 모델 로드
        model, config, device = load_model_and_config(args)
        print(f"Model loaded successfully on {device}")
        
        # 2. 데이터로더 생성
        dataloader = create_dummy_dataloader(args.data_path, args.max_samples, args.batch_size)
        print(f"Dataloader created with {len(dataloader)} batches")
        
        # 3. Gate 추적과 함께 추론 실행
        results = run_inference_with_gate_tracking(model, dataloader, device, output_dir)
        
        if results is None:
            print("Gate tracking is not supported by this model.")
            return
        
        # 4. Gate 값 분석
        analyze_gate_values(model, output_dir, args)
        
        # 5. 설정 정보 저장
        config_info = {
            'timestamp': timestamp,
            'args': vars(args),
            'config': config,
            'device': str(device),
            'total_samples': len(results),
            'gating_type': args.gating_type,
            'model_path': args.model_path,
            'data_path': args.data_path
        }
        
        with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print("="*50)
        print("Gate analysis completed successfully!")
        print(f"Results saved to: {output_dir}")
        print("="*50)
        
        # 빠른 요약 출력
        if hasattr(model, 'avigate_fusion'):
            model.avigate_fusion.print_gate_summary()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
