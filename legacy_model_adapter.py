#!/usr/bin/env python3
"""
Legacy Model Adapter for Gate Analysis

이미 훈련된 모델(gate 추적 기능이 없는 모델)을 
새로운 gate 추적 기능과 호환되도록 변환하는 어댑터입니다.

Usage:
    from legacy_model_adapter import adapt_legacy_model
    
    # 기존 모델 로드
    model = torch.load('old_model.pth')
    
    # Gate 추적 기능 추가
    adapted_model = adapt_legacy_model(model)
    adapted_model.avigate_fusion.enable_gate_tracking()
"""

import torch
import torch.nn as nn
import copy
from qd_detr.avigate_custom import AVIGATEFusionCustom, GatedFusionBlockCustom, GatingFunction

def adapt_legacy_model(model, gating_type='global'):
    """
    기존 모델을 gate 추적 기능이 있는 모델로 변환합니다.
    
    Args:
        model: 기존 훈련된 모델
        gating_type: gate 타입 ('global', 'clipwise', 'elementwise')
    
    Returns:
        gate 추적 기능이 추가된 모델
    """
    print("Adapting legacy model for gate tracking...")
    
    # 1. 모델에서 AVIGATEFusion 관련 부분 찾기
    avigate_fusion = find_avigate_fusion_module(model)
    
    if avigate_fusion is None:
        print("Warning: No AVIGATEFusion module found in the model")
        return model
    
    # 2. 기존 가중치 추출
    old_state_dict = avigate_fusion.state_dict()
    
    # 3. 새로운 AVIGATEFusionCustom 생성
    new_avigate_fusion = create_new_avigate_fusion(avigate_fusion, gating_type)
    
    # 4. 가중치 복사
    transfer_weights(old_state_dict, new_avigate_fusion)
    
    # 5. 모델에서 교체
    replace_avigate_fusion_in_model(model, new_avigate_fusion)
    
    print(f"Model adapted successfully with {gating_type} gating")
    return model

def find_avigate_fusion_module(model):
    """모델에서 AVIGATEFusion 모듈을 찾습니다."""
    # 가능한 속성 이름들
    possible_names = [
        'avigate_fusion',
        'fusion_module', 
        'multimodal_fusion',
        'audio_video_fusion'
    ]
    
    for name in possible_names:
        if hasattr(model, name):
            module = getattr(model, name)
            print(f"Found fusion module: {name}")
            return module
    
    # 재귀적으로 하위 모듈 검색
    for name, module in model.named_modules():
        if 'avigate' in name.lower() or 'fusion' in name.lower():
            print(f"Found potential fusion module: {name}")
            return module
    
    return None

def create_new_avigate_fusion(old_module, gating_type):
    """새로운 gate 추적 기능이 있는 AVIGATEFusion을 생성합니다."""
    # 기존 모듈의 설정 추출
    config = extract_module_config(old_module)
    
    # 새로운 모듈 생성
    new_module = AVIGATEFusionCustom(
        vid_dim=config['vid_dim'],
        aud_dim=config['aud_dim'], 
        hidden_dim=config['hidden_dim'],
        n_heads=config['n_heads'],
        num_layers=config['num_layers'],
        gating_type=gating_type
    )
    
    return new_module

def extract_module_config(module):
    """기존 모듈에서 설정을 추출합니다."""
    config = {
        'vid_dim': 256,  # 기본값
        'aud_dim': 256,
        'hidden_dim': 256,
        'n_heads': 8,
        'num_layers': 1
    }
    
    try:
        # fusion_layers에서 설정 추출
        if hasattr(module, 'fusion_layers') and len(module.fusion_layers) > 0:
            first_layer = module.fusion_layers[0]
            config['num_layers'] = len(module.fusion_layers)
            
            if hasattr(first_layer, 'n_heads'):
                config['n_heads'] = first_layer.n_heads
                
            if hasattr(first_layer, 'head_dim'):
                config['hidden_dim'] = first_layer.n_heads * first_layer.head_dim
            
            # Linear layer에서 차원 추출
            if hasattr(first_layer, 'a_proj'):
                config['hidden_dim'] = first_layer.a_proj.in_features
                config['vid_dim'] = config['hidden_dim']
                config['aud_dim'] = config['hidden_dim']
        
        print(f"Extracted config: {config}")
        
    except Exception as e:
        print(f"Warning: Could not extract full config, using defaults: {e}")
    
    return config

def transfer_weights(old_state_dict, new_module):
    """기존 가중치를 새로운 모듈로 전송합니다."""
    new_state_dict = new_module.state_dict()
    
    # 가중치 매핑
    transferred_keys = []
    skipped_keys = []
    
    for old_key, old_weight in old_state_dict.items():
        # Gate tracking 관련 키는 건너뛰기
        if any(skip_pattern in old_key for skip_pattern in [
            'track_gates', 'gate_history', 'current_sample_id'
        ]):
            continue
            
        if old_key in new_state_dict:
            if old_weight.shape == new_state_dict[old_key].shape:
                new_state_dict[old_key] = old_weight
                transferred_keys.append(old_key)
            else:
                print(f"Shape mismatch for {old_key}: {old_weight.shape} vs {new_state_dict[old_key].shape}")
                skipped_keys.append(old_key)
        else:
            # 키 이름 매핑 시도
            mapped_key = try_key_mapping(old_key, new_state_dict.keys())
            if mapped_key and old_weight.shape == new_state_dict[mapped_key].shape:
                new_state_dict[mapped_key] = old_weight
                transferred_keys.append(f"{old_key} -> {mapped_key}")
            else:
                skipped_keys.append(old_key)
    
    # 새로운 state dict 로드
    new_module.load_state_dict(new_state_dict, strict=False)
    
    print(f"Transferred {len(transferred_keys)} weights")
    print(f"Skipped {len(skipped_keys)} weights")
    
    if skipped_keys:
        print("Skipped keys:", skipped_keys[:5], "..." if len(skipped_keys) > 5 else "")

def try_key_mapping(old_key, new_keys):
    """이전 키를 새로운 키로 매핑 시도"""
    # 간단한 매핑 규칙들
    mapping_rules = [
        # 예: 'module.fusion.layer.0.weight' -> 'fusion_layers.0.weight'
        lambda k: k.replace('module.fusion.layer.', 'fusion_layers.'),
        lambda k: k.replace('fusion.layer.', 'fusion_layers.'),
        lambda k: k.replace('avigate.', ''),
    ]
    
    for rule in mapping_rules:
        candidate = rule(old_key)
        if candidate in new_keys:
            return candidate
    
    return None

def replace_avigate_fusion_in_model(model, new_avigate_fusion):
    """모델에서 기존 fusion 모듈을 새로운 것으로 교체합니다."""
    # 직접 속성 교체
    if hasattr(model, 'avigate_fusion'):
        model.avigate_fusion = new_avigate_fusion
        return
    
    # 다른 이름으로 된 fusion 모듈 찾아서 교체
    fusion_names = ['fusion_module', 'multimodal_fusion', 'audio_video_fusion']
    for name in fusion_names:
        if hasattr(model, name):
            setattr(model, name, new_avigate_fusion)
            # 별칭도 생성
            model.avigate_fusion = new_avigate_fusion
            return
    
    # 하위 모듈에서 교체
    for name, module in model.named_modules():
        if 'fusion' in name.lower():
            parent_names = name.split('.')[:-1]
            attr_name = name.split('.')[-1]
            
            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)
            
            setattr(parent, attr_name, new_avigate_fusion)
            # 모델 최상위에 별칭 생성
            model.avigate_fusion = new_avigate_fusion
            return
    
    print("Warning: Could not replace fusion module, adding as new attribute")
    model.avigate_fusion = new_avigate_fusion

def load_legacy_model_with_gate_tracking(model_path, gating_type='global', device='auto'):
    """
    기존 모델을 로드하고 gate 추적 기능을 추가합니다.
    
    Args:
        model_path: 모델 체크포인트 경로
        gating_type: gate 타입
        device: 디바이스 ('auto', 'cpu', 'cuda')
    
    Returns:
        gate 추적 기능이 추가된 모델
    """
    print(f"Loading legacy model from {model_path}")
    
    # 디바이스 설정
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    # 체크포인트 구조 확인
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint
    
    # 모델 구조 재구성 (실제 프로젝트에 맞게 수정 필요)
    try:
        from qd_detr.model import build_model
        from qd_detr.config import BaseOptions
        
        # 기본 설정으로 모델 빌드
        config = {
            'use_avigate_custom': True,
            'gating_type': gating_type,
            'hidden_dim': 256,
            'nheads': 8,
            'num_queries': 10
        }
        
        model = build_model(config)
        model.load_state_dict(model_state, strict=False)
        
    except Exception as e:
        print(f"Warning: Could not build model with project function: {e}")
        print("Creating dummy model structure...")
        
        # 더미 모델 생성 (실제 사용 시 프로젝트 구조에 맞게 수정)
        model = create_dummy_model_structure(model_state, gating_type)
    
    # Gate 추적 기능 추가
    model = adapt_legacy_model(model, gating_type)
    
    model.eval()
    model.to(device)
    
    print(f"Legacy model loaded successfully on {device}")
    return model

def create_dummy_model_structure(state_dict, gating_type):
    """상태 딕셔너리에서 더미 모델 구조 생성"""
    class LegacyModel(nn.Module):
        def __init__(self, gating_type):
            super().__init__()
            # 기본 AVIGATEFusion 구조 생성
            self.avigate_fusion = AVIGATEFusionCustom(
                vid_dim=256, aud_dim=256, hidden_dim=256,
                n_heads=8, num_layers=1, gating_type=gating_type
            )
            
        def forward(self, batch):
            # 더미 forward (실제 사용 시 수정 필요)
            bs = batch.get('batch_size', 1)
            seq_len = batch.get('seq_len', 75)
            
            video_feat = torch.randn(bs, seq_len, 256)
            audio_feat = torch.randn(bs, seq_len, 256)
            
            fused_feat = self.avigate_fusion(video_feat, audio_feat)
            return {'predictions': fused_feat}
    
    return LegacyModel(gating_type)

# 사용 예시 함수들
def quick_test_legacy_model(model_path, gating_type='global'):
    """기존 모델로 빠른 gate 추적 테스트"""
    try:
        # 모델 로드 및 적응
        model = load_legacy_model_with_gate_tracking(model_path, gating_type)
        
        # Gate 추적 활성화
        model.avigate_fusion.enable_gate_tracking()
        
        # 테스트 추론
        for i in range(3):
            model.avigate_fusion.set_current_sample_id(f'legacy_test_{i}')
            
            # 더미 데이터
            batch = {
                'batch_size': 1,
                'seq_len': 75
            }
            
            with torch.no_grad():
                outputs = model(batch)
        
        # 결과 확인
        model.avigate_fusion.print_gate_summary()
        
        print("Legacy model gate tracking test successful!")
        return True
        
    except Exception as e:
        print(f"Legacy model test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test legacy model adaptation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to legacy model checkpoint")
    parser.add_argument("--gating_type", type=str, default="global",
                       choices=["global", "clipwise", "elementwise"])
    
    args = parser.parse_args()
    
    success = quick_test_legacy_model(args.model_path, args.gating_type)
    if success:
        print("✅ Legacy model adaptation successful!")
    else:
        print("❌ Legacy model adaptation failed!")
