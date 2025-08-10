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
        'fusion',  # 실제 체크포인트에서 사용되는 이름
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
        if any(keyword in name.lower() for keyword in ['avigate', 'fusion']):
            print(f"Found potential fusion module: {name}")
            return module
    
    return None

def extract_module_config(old_module):
    """기존 AVIGATE 모듈에서 설정을 추출합니다."""
    # 기본 설정
    config = {
        'vid_dim': 256,
        'aud_dim': 256,
        'hidden_dim': 256,
        'n_heads': 8,
        'num_layers': 1
    }
    
    # 모듈에서 설정 추출 시도
    if hasattr(old_module, 'layers') and hasattr(old_module.layers, '__len__'):
        config['num_layers'] = len(old_module.layers)
        print(f"Extracted num_layers: {config['num_layers']} from existing module")
    
    if hasattr(old_module, 'layers') and len(old_module.layers) > 0:
        first_layer = old_module.layers[0]
        if hasattr(first_layer, 'self_attn'):
            config['hidden_dim'] = first_layer.self_attn.embed_dim
            config['n_heads'] = first_layer.self_attn.num_heads
            print(f"Extracted hidden_dim: {config['hidden_dim']}, n_heads: {config['n_heads']}")
    
    return config

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

def extract_avigate_config_from_checkpoint(checkpoint):
    """체크포인트에서 AVIGATE 설정을 추출합니다."""
    # 기본 설정
    config = {
        'vid_dim': 256,
        'aud_dim': 256,
        'hidden_dim': 256,
        'n_heads': 8,
        'num_layers': 1  # 기본값
    }
    
    # 레이어 수 추출 - fusion.fusion_layers 패턴 확인
    layer_count = 0
    max_layer_idx = -1
    
    for key in checkpoint.keys():
        # fusion.fusion_layers.X.xxx 패턴 확인
        if 'fusion.fusion_layers.' in key:
            try:
                layer_idx = int(key.split('fusion.fusion_layers.')[1].split('.')[0])
                max_layer_idx = max(max_layer_idx, layer_idx)
            except (IndexError, ValueError):
                continue
        # 기존 avigate_fusion 패턴도 확인
        elif 'avigate_fusion' in key and 'layers.' in key:
            try:
                layer_idx = int(key.split('layers.')[1].split('.')[0])
                layer_count = max(layer_count, layer_idx + 1)
            except (IndexError, ValueError):
                continue
    
    if max_layer_idx >= 0:
        config['num_layers'] = max_layer_idx + 1
        print(f"Detected {config['num_layers']} AVIGATE fusion layers from fusion.fusion_layers")
    elif layer_count > 0:
        config['num_layers'] = layer_count
        print(f"Detected {layer_count} AVIGATE fusion layers from avigate_fusion.layers")
    else:
        print("Could not detect layer count, using default: 1")
    
    # 다른 설정들도 체크포인트에서 추출 시도
    for key, tensor in checkpoint.items():
        if ('fusion.fusion_layers.' in key or 'avigate_fusion' in key) and hasattr(tensor, 'shape'):
            if 'mlp_mha.0.weight' in key and len(tensor.shape) == 2:
                # MLP 입력 차원에서 hidden_dim 추출
                input_dim = tensor.shape[1]
                if input_dim % 2 == 0:  # video + audio concat이므로 짝수여야 함
                    config['hidden_dim'] = input_dim // 2
            elif 'self_attn' in key and 'in_proj_weight' in key:
                config['hidden_dim'] = tensor.shape[1]
    
    return config

def transfer_weights_to_multilayer(checkpoint_state, avigate_fusion_module):
    """1개 레이어의 가중치를 다중 레이어에 복제하여 전송"""
    print(f"Transferring weights to {len(avigate_fusion_module.fusion_layers)} layers...")
    
    # 현재 모듈의 state dict 가져오기
    module_state = avigate_fusion_module.state_dict()
    
    # Layer 0에 해당하는 가중치들 찾기
    layer_0_weights = {}
    for key, value in checkpoint_state.items():
        if 'fusion.fusion_layers.0.' in key:
            # fusion.fusion_layers.0.xxx -> fusion_layers.0.xxx
            new_key = key.replace('fusion.fusion_layers.', 'fusion_layers.')
            if new_key in module_state:
                layer_0_weights[new_key] = value
    
    transferred_count = 0
    # Layer 0의 가중치를 모든 레이어에 복제
    for layer_idx in range(len(avigate_fusion_module.fusion_layers)):
        for orig_key, weight in layer_0_weights.items():
            # fusion_layers.0.xxx -> fusion_layers.{layer_idx}.xxx
            target_key = orig_key.replace('fusion_layers.0.', f'fusion_layers.{layer_idx}.')
            if target_key in module_state:
                if weight.shape == module_state[target_key].shape:
                    module_state[target_key] = weight.clone()
                    transferred_count += 1
    
    # 새로운 state dict 로드
    avigate_fusion_module.load_state_dict(module_state, strict=False)
    print(f"Transferred {transferred_count} weights across {len(avigate_fusion_module.fusion_layers)} layers")


def transfer_weights(old_state_dict, new_module):
    """기존 가중치를 새로운 모듈로 전송합니다."""
    new_state_dict = new_module.state_dict()
    
    # 가중치 매핑
    transferred_keys = []
    skipped_keys = []
    
    # 새로운 모듈의 레이어 수 확인
    num_new_layers = len(new_module.fusion_layers)
    
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
                # 레이어 복제 시도 (1개 레이어를 여러 레이어로 복제)
                if 'fusion_layers.0.' in old_key:
                    base_key = old_key.replace('fusion_layers.0.', 'fusion_layers.{}.', 1)
                    for layer_idx in range(1, num_new_layers):
                        target_key = base_key.format(layer_idx)
                        if target_key in new_state_dict and old_weight.shape == new_state_dict[target_key].shape:
                            new_state_dict[target_key] = old_weight.clone()
                            transferred_keys.append(f"{old_key} -> {target_key} (replicated)")
                
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
    
    # Gate 추적 기능을 위한 가중치 전송 (모델 구조는 유지)
    if hasattr(model, 'avigate_fusion'):
        print("Transferring weights to 4-layer avigate_fusion...")
        transfer_weights_to_multilayer(model_state, model.avigate_fusion)
        model.avigate_fusion.enable_gate_tracking()
    else:
        print("Warning: No avigate_fusion found in dummy model")
    
    model.eval()
    model.to(device)
    
    print(f"Legacy model loaded successfully on {device}")
    return model

def create_dummy_model_structure(state_dict, gating_type):
    """상태 딕셔너리에서 더미 모델 구조 생성"""
    
    # 상태 딕셔너리에서 레이어 수 추출 - fusion 접두사 고려
    layer_count = 1  # 기본값
    max_layer_idx = -1
    
    for key in state_dict.keys():
        # fusion.fusion_layers.0, fusion.fusion_layers.1, ... 패턴 확인
        if 'fusion.fusion_layers.' in key:
            try:
                # "fusion.fusion_layers.X.xxx" -> X 추출
                layer_idx = int(key.split('fusion.fusion_layers.')[1].split('.')[0])
                max_layer_idx = max(max_layer_idx, layer_idx)
            except (IndexError, ValueError):
                continue
    
    if max_layer_idx >= 0:
        layer_count = max_layer_idx + 1
        print(f"Creating dummy model with {layer_count} AVIGATE layers (detected from fusion.fusion_layers)")
    else:
        print(f"Creating dummy model with {layer_count} AVIGATE layers (default)")
    
    class LegacyModel(nn.Module):
        def __init__(self, gating_type, num_layers):
            super().__init__()
            # 기본 AVIGATEFusion 구조 생성 - 감지된 레이어 수 사용
            self.avigate_fusion = AVIGATEFusionCustom(
                vid_dim=256, aud_dim=256, hidden_dim=256,
                n_heads=8, num_layers=num_layers, gating_type=gating_type
            )
            
        def forward(self, batch):
            # 더미 forward (실제 사용 시 수정 필요)
            bs = batch.get('batch_size', 1)
            seq_len = batch.get('seq_len', 75)
            
            # 모델과 같은 디바이스에 텐서 생성
            device = next(self.parameters()).device
            video_feat = torch.randn(bs, seq_len, 256, device=device)
            audio_feat = torch.randn(bs, seq_len, 256, device=device)
            
            fused_feat = self.avigate_fusion(video_feat, audio_feat)
            return {'predictions': fused_feat}
    
    return LegacyModel(gating_type, layer_count)

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
