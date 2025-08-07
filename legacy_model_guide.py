#!/usr/bin/env python3
"""
Legacy Model Gate Analysis Guide

이미 훈련된 모델(gate 추적 기능이 없는)로 gate 분석을 하는 방법을 안내합니다.

기존 모델의 가중치는 그대로 유지하면서 gate 추적 기능만 추가합니다.
"""

import os
import sys

def print_usage_guide():
    """사용법 가이드를 출력합니다."""
    print("=" * 60)
    print("🔄 Legacy Model Gate Analysis Guide")
    print("=" * 60)
    
    print("\n📋 상황 설명:")
    print("- 이미 훈련된 모델이 있음 (gate 추적 기능 없음)")
    print("- 새로운 gate 추적 기능으로 분석하고 싶음")
    print("- 모델 가중치는 그대로 유지하면서 기능만 추가")
    
    print("\n🔧 해결 방법:")
    print("1. Legacy Model Adapter 사용")
    print("2. 기존 가중치를 새로운 구조로 전송") 
    print("3. Gate 추적 기능 활성화")
    
    print("\n📝 사용법:")
    
    print("\n🎯 방법 1: 명령행에서 실행")
    print("bash")
    print("python run_gate_analysis.py \\")
    print("    --model_path /path/to/your/trained_model.pth \\")
    print("    --data_path /path/to/your/data.json \\")
    print("    --legacy_model \\")  # 중요: 이 플래그 추가
    print("    --gating_type global \\")
    print("    --max_samples 50 \\")
    print("    --compare_samples \\")
    print("    --save_raw_data")
    
    print("\n🎯 방법 2: Python 코드에서 직접 사용")
    print("""
```python
from legacy_model_adapter import load_legacy_model_with_gate_tracking
from gate_analysis_utils import GateAnalyzer

# 1. 레거시 모델 로드 (gate 추적 기능 자동 추가)
model = load_legacy_model_with_gate_tracking(
    model_path='path/to/your/model.pth',
    gating_type='global'  # 'global', 'clipwise', 'elementwise' 중 선택
)

# 2. Gate 추적 활성화
model.avigate_fusion.enable_gate_tracking()

# 3. 추론 및 gate 값 수집
for batch_idx, batch in enumerate(dataloader):
    model.avigate_fusion.set_current_sample_id(f"sample_{batch_idx}")
    
    with torch.no_grad():
        outputs = model(batch)

# 4. Gate 분석
analyzer = GateAnalyzer(model.avigate_fusion)
analyzer.analyze_all_samples("./legacy_gate_analysis")

# 5. 간단한 요약 확인
model.avigate_fusion.print_gate_summary()
```
""")

    print("\n🎯 방법 3: 기존 모델을 새 구조로 변환")
    print("""
```python
from legacy_model_adapter import adapt_legacy_model

# 기존 방식으로 모델 로드
model = torch.load('your_model.pth')

# Gate 추적 기능 추가
adapted_model = adapt_legacy_model(model, gating_type='global')

# 이제 gate 추적 가능
adapted_model.avigate_fusion.enable_gate_tracking()
```
""")

    print("\n⚠️  주의사항:")
    print("- 가중치는 그대로 유지됩니다 (성능 변화 없음)")
    print("- gating_type은 추론시에만 영향을 줍니다")
    print("- 첫 실행시 가중치 변환에 시간이 걸릴 수 있습니다")
    print("- GPU 메모리 사용량이 약간 증가할 수 있습니다")
    
    print("\n🚀 빠른 테스트:")
    print("python test_legacy_model.py --model_path your_model.pth")
    
    print("\n📊 분석 결과:")
    print("- gate_analysis_결과/: 전체 분석 결과")
    print("- ANALYSIS_REPORT.md: 종합 리포트")
    print("- individual_samples/: 샘플별 상세 분석")
    print("- raw_data/: 원시 gate 값 (옵션)")

def create_test_script():
    """레거시 모델 테스트 스크립트를 생성합니다."""
    test_script = """#!/usr/bin/env python3
\"\"\"
Legacy Model Quick Test

기존 훈련된 모델로 gate 분석이 가능한지 빠르게 테스트합니다.
\"\"\"

import argparse
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_legacy_model(model_path, gating_type='global'):
    try:
        from legacy_model_adapter import load_legacy_model_with_gate_tracking
        
        print(f"Loading model: {model_path}")
        print(f"Gating type: {gating_type}")
        
        # 모델 로드
        model = load_legacy_model_with_gate_tracking(model_path, gating_type)
        
        # Gate 추적 활성화
        model.avigate_fusion.enable_gate_tracking()
        print("✅ Gate tracking enabled successfully")
        
        # 테스트 추론
        print("Running test inference...")
        for i in range(3):
            model.avigate_fusion.set_current_sample_id(f'test_{i}')
            
            # 더미 배치 (실제 데이터 구조에 맞게 수정)
            batch = {
                'batch_size': 1,
                'seq_len': 75
            }
            
            with torch.no_grad():
                outputs = model(batch)
        
        print("✅ Test inference completed")
        
        # Gate 값 확인
        model.avigate_fusion.print_gate_summary()
        
        print("\\n🎉 Legacy model gate analysis is ready!")
        print("You can now use this model for full gate analysis.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to legacy model")
    parser.add_argument("--gating_type", default="global", 
                       choices=["global", "clipwise", "elementwise"])
    
    args = parser.parse_args()
    
    success = test_legacy_model(args.model_path, args.gating_type)
    
    if success:
        print("\\n✅ Ready for full analysis! Run:")
        print(f"python run_gate_analysis.py --model_path {args.model_path} --legacy_model --data_path your_data.json")
    else:
        print("\\n❌ Please check the error messages above.")
"""

    with open('/workspace/QD-DETR-exp/test_legacy_model.py', 'w') as f:
        f.write(test_script)
    
    # 실행 권한 부여
    os.chmod('/workspace/QD-DETR-exp/test_legacy_model.py', 0o755)
    print("✅ test_legacy_model.py created")

def show_example_commands():
    """실제 사용 예시 명령어들을 보여줍니다."""
    print("\n📋 실제 사용 예시:")
    
    print("\n1️⃣ 레거시 모델 호환성 테스트:")
    print("python test_legacy_model.py --model_path /path/to/your/model.pth")
    
    print("\n2️⃣ Global gating으로 분석:")
    print("""python run_gate_analysis.py \\
    --model_path /path/to/your/model.pth \\
    --data_path /path/to/your/test_data.json \\
    --legacy_model \\
    --gating_type global \\
    --max_samples 100 \\
    --compare_samples""")
    
    print("\n3️⃣ Clipwise gating으로 상세 분석:")
    print("""python run_gate_analysis.py \\
    --model_path /path/to/your/model.pth \\
    --data_path /path/to/your/test_data.json \\
    --legacy_model \\
    --gating_type clipwise \\
    --max_samples 50 \\
    --save_raw_data \\
    --output_dir ./detailed_gate_analysis""")
    
    print("\n4️⃣ 특정 샘플들만 분석:")
    print("""python run_gate_analysis.py \\
    --model_path /path/to/your/model.pth \\
    --data_path /path/to/your/test_data.json \\
    --legacy_model \\
    --analyze_samples video_123 video_456 video_789""")

def main():
    print_usage_guide()
    create_test_script()
    show_example_commands()
    
    print("\n" + "=" * 60)
    print("🚀 Ready to analyze your legacy model!")
    print("=" * 60)
    
    # 사용자 입력 받기
    choice = input("\n원하는 작업을 선택하세요:\n1. 모델 호환성 테스트\n2. 전체 gate 분석\n3. 도움말 다시 보기\n선택 (1-3): ").strip()
    
    if choice == "1":
        model_path = input("모델 경로를 입력하세요: ").strip()
        if os.path.exists(model_path):
            os.system(f"python test_legacy_model.py --model_path {model_path}")
        else:
            print("❌ 모델 파일을 찾을 수 없습니다.")
    
    elif choice == "2":
        model_path = input("모델 경로를 입력하세요: ").strip()
        data_path = input("데이터 경로를 입력하세요: ").strip()
        gating_type = input("Gating type (global/clipwise/elementwise, 기본값: global): ").strip() or "global"
        
        if os.path.exists(model_path):
            cmd = f"python run_gate_analysis.py --model_path {model_path} --data_path {data_path} --legacy_model --gating_type {gating_type} --max_samples 20"
            print(f"실행 명령어: {cmd}")
            
            confirm = input("실행하시겠습니까? (y/n): ").strip().lower()
            if confirm == 'y':
                os.system(cmd)
        else:
            print("❌ 모델 파일을 찾을 수 없습니다.")
    
    elif choice == "3":
        main()  # 재귀 호출
    
    else:
        print("올바른 선택지를 입력해주세요.")

if __name__ == "__main__":
    main()
