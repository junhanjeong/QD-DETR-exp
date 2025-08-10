# AVIGATEFusionCustom Gate Analysis System

AVIGATEFusionCustom 모듈의 gate 값을 추적하고 분석하는 종합적인 시스템입니다. 이 시스템을 통해 추론 과정에서 각 쿼리마다 attention과 FFN gate 값의 변화를 실시간으로 모니터링하고, 상세한 분석을 수행할 수 있습니다.

**✨ 특별 기능: 이미 훈련된 모델(레거시 모델)에서도 gate 분석이 가능합니다!**

## 🎯 주요 기능

- **실시간 Gate 추적**: 추론 과정에서 각 레이어의 attention과 FFN gate 값을 실시간으로 저장
- **레거시 모델 지원**: 이미 훈련된 모델(gate 추적 기능 없음)에서도 gate 분석 가능
- **다양한 Gating Type 지원**: Global, Clipwise, Elementwise gating 타입별 분석
- **종합적인 통계 분석**: 평균, 표준편차, 분포, 백분위수 등 상세 통계
- **시각화**: 히트맵, 분포도, 진화 패턴, 샘플 비교 등 다양한 시각화
- **샘플 비교**: 여러 샘플 간의 gate 활성화 패턴 비교
- **자동 리포트 생성**: 분석 결과를 종합한 자동 리포트 생성

## 📁 파일 구조

```
QD-DETR-exp/
├── qd_detr/
│   └── avigate_custom.py          # 수정된 AVIGATEFusionCustom 모듈
├── gate_analysis_utils.py         # Gate 분석 유틸리티 클래스
├── run_gate_analysis.py          # 실제 추론에서 gate 분석 실행
├── test_gate_analysis.py         # Gate 분석 시스템 테스트
├── legacy_model_adapter.py       # 레거시 모델 어댑터 (★ 새로운 기능)
├── legacy_model_guide.py         # 레거시 모델 사용 가이드
├── test_legacy_model.py          # 레거시 모델 호환성 테스트
└── GATE_ANALYSIS_README.md       # 이 파일
```

## 🚀 빠른 시작

### 0. 레거시 모델 사용 (이미 훈련된 모델이 있는 경우) ⭐

```bash
# 레거시 모델 호환성 가이드 실행
python legacy_model_guide.py

# 또는 직접 테스트
python test_legacy_model.py --model_path /path/to/your/trained_model.pth

# 레거시 모델로 gate 분석
python run_gate_analysis.py \
    --model_path /path/to/your/trained_model.pth \
    --data_path /path/to/your/data.json \
    --legacy_model \
    --gating_type global \
    --max_samples 50
```

### 2. 테스트 실행 (실제 모델 없이)

```bash
# 빠른 데모
python test_gate_analysis.py --quick

# 종합 테스트
python test_gate_analysis.py --comprehensive
```

### 3. 실제 모델에서 gate 분석

```python
# 모델에서 gate 추적 활성화
model.avigate_fusion.enable_gate_tracking()

# 추론 중 샘플 ID 설정
for batch_idx, batch in enumerate(dataloader):
    model.avigate_fusion.set_current_sample_id(f"sample_{batch_idx}")
    outputs = model(batch)

# 분석 실행
from gate_analysis_utils import GateAnalyzer
analyzer = GateAnalyzer(model.avigate_fusion)
analyzer.analyze_all_samples("./gate_analysis_results")
```

### 4. 명령행에서 분석 실행

```bash
# 새로운 모델 (gate 추적 기능 내장)
python run_gate_analysis.py \
    --model_path /path/to/model.pth \
    --data_path /path/to/data.json \
    --max_samples 50 \
    --gating_type global \
    --compare_samples \
    --save_raw_data

# 레거시 모델 (이미 훈련된 모델)
python run_gate_analysis.py \
    --model_path /path/to/legacy_model.pth \
    --data_path /path/to/data.json \
    --legacy_model \
    --gating_type global \
    --max_samples 50 \
    --compare_samples
```

## � 레거시 모델 지원 (이미 훈련된 모델)

### 📋 상황 설명

기존에 훈련된 모델들은 gate 추적 기능이 없어서 직접적으로 gate 분석이 불가능했습니다. 하지만 **Legacy Model Adapter**를 통해 기존 모델의 가중치를 그대로 유지하면서 gate 추적 기능만 추가할 수 있습니다.

### 🔧 Legacy Model Adapter 사용법

#### 방법 1: 자동 변환 및 분석

```python
from legacy_model_adapter import load_legacy_model_with_gate_tracking

# 기존 모델을 gate 추적 기능과 함께 로드
model = load_legacy_model_with_gate_tracking(
    model_path='path/to/your/trained_model.pth',
    gating_type='global'  # 'global', 'clipwise', 'elementwise' 중 선택
)

# 즉시 gate 추적 가능
model.avigate_fusion.enable_gate_tracking()

# 기존과 동일한 방식으로 추론
for batch in dataloader:
    model.avigate_fusion.set_current_sample_id(f"sample_{batch['qid']}")
    outputs = model(batch)

# Gate 분석
from gate_analysis_utils import GateAnalyzer
analyzer = GateAnalyzer(model.avigate_fusion)
analyzer.analyze_all_samples("./legacy_analysis_results")
```

#### 방법 2: 수동 변환

```python
from legacy_model_adapter import adapt_legacy_model

# 기존 방식으로 모델 로드
checkpoint = torch.load('your_model.pth')
model = YourModelClass()
model.load_state_dict(checkpoint['model'])

# Gate 추적 기능 추가
adapted_model = adapt_legacy_model(model, gating_type='global')

# 이제 gate 추적 가능
adapted_model.avigate_fusion.enable_gate_tracking()
```

### 🚀 빠른 시작 (레거시 모델)

```bash
# 1. 호환성 테스트
python test_legacy_model.py --model_path your_model.pth

# 2. 대화형 가이드 실행
python legacy_model_guide.py

# 3. 직접 분석 실행
python run_gate_analysis.py \
    --model_path your_model.pth \
    --data_path your_data.json \
    --legacy_model \
    --gating_type global
```

### ⚡ 중요 특징

- **가중치 보존**: 기존 모델의 성능이 완전히 보존됩니다
- **즉시 사용**: 추가 훈련 없이 바로 gate 분석 가능
- **유연한 Gating**: 추론 시점에 gating type 선택 가능
- **백워드 호환**: 기존 코드 구조와 완전 호환

## �🔧 API 사용법

### AVIGATEFusionCustom 메서드

```python
# Gate 추적 활성화/비활성화
model.avigate_fusion.enable_gate_tracking()
model.avigate_fusion.disable_gate_tracking()

# 현재 샘플 ID 설정
model.avigate_fusion.set_current_sample_id("sample_001")

# Gate 통계 조회
stats = model.avigate_fusion.get_gate_statistics()
stats_single = model.avigate_fusion.get_gate_statistics("sample_001")

# 간단한 요약 출력
model.avigate_fusion.print_gate_summary()

# 개별 샘플 분석 저장
model.avigate_fusion.save_gate_analysis("./output_dir", "sample_001")

# 여러 샘플 비교
sample_ids = ["sample_001", "sample_002", "sample_003"]
model.avigate_fusion.compare_samples_gates(sample_ids, "./comparison_dir")
```

### GateAnalyzer 클래스

```python
from gate_analysis_utils import GateAnalyzer

# 분석기 초기화
analyzer = GateAnalyzer(model.avigate_fusion)

# 전체 분석 실행
analyzer.analyze_all_samples("./analysis_results")
```

## 📊 출력 파일 설명

### 분석 결과 디렉토리 구조

```
gate_analysis_results/
├── ANALYSIS_REPORT.md                    # 종합 분석 리포트
├── overall_statistics.json              # 전체 통계 (JSON)
├── overall_statistics.png               # 전체 통계 시각화
├── layer_distributions.png              # 레이어별 분포
├── activation_patterns.png              # 활성화 패턴 히트맵
├── gate_clustering.png                  # 클러스터링 분석 (옵션)
├── individual_samples/                   # 개별 샘플 분석
│   ├── sample_001_details.json
│   ├── gate_analysis_sample_001.png
│   └── ...
├── comparisons/                          # 샘플 비교 결과
│   └── gate_comparison.png
└── raw_data/                            # 원시 gate 값 (옵션)
    ├── sample_001_layer0_att.npy
    ├── sample_001_layer0_ffn.npy
    └── ...
```

### 주요 시각화

1. **overall_statistics.png**: 전체 gate 분포, 레이어별 평균/분산
2. **layer_distributions.png**: 각 레이어별 상세 gate 분포
3. **activation_patterns.png**: 샘플 x 레이어 히트맵
4. **gate_analysis_[sample_id].png**: 개별 샘플의 gate 값 시각화
5. **gate_comparison.png**: 여러 샘플의 gate 값 비교

## 📈 Gate 값 해석 가이드

### Gate 값의 의미

- **Range**: Gate 값은 tanh 함수를 통해 [-1, 1] 범위
- **Positive values**: Audio 정보가 video에 더해짐
- **Negative values**: Audio 정보가 억제됨
- **Zero**: Audio 정보가 무시됨

### Gating Type별 특성

#### Global Gating
- **Shape**: (batch_size, 1, 1)
- **의미**: 전체 시퀀스에 대해 동일한 gate 값
- **사용 시기**: 전반적인 audio-video 관련성이 일정할 때

#### Clipwise Gating  
- **Shape**: (batch_size, seq_len, 1)
- **의미**: 각 시간 단계별로 다른 gate 값
- **사용 시기**: 시간에 따라 audio-video 관련성이 변할 때

#### Elementwise Gating
- **Shape**: (batch_size, seq_len, hidden_dim)
- **의미**: 각 특성 차원별로 다른 gate 값
- **사용 시기**: 세밀한 특성별 제어가 필요할 때

### 분석 포인트

1. **Gate 활성화 수준**
   - 평균 gate 값이 0에 가까우면 audio 정보가 거의 사용되지 않음
   - 평균 gate 값이 양수면 audio 정보가 적극적으로 활용됨

2. **Gate 변동성**
   - 표준편차가 클수록 다양한 상황에서 다르게 반응
   - 표준편차가 작을수록 일관된 반응

3. **레이어별 패턴**
   - 초기 레이어: 저수준 특성 융합
   - 후기 레이어: 고수준 의미 융합

4. **샘플별 차이**
   - 샘플마다 다른 gate 패턴은 모델이 입력에 따라 적응적으로 반응함을 의미

## 🛠 커스터마이징

### 추가 분석 메트릭

```python
# 커스텀 분석 함수 추가
def custom_gate_analysis(model):
    gate_history = model.avigate_fusion.gate_history
    
    # 예: Gate 값의 엔트로피 계산
    for sample_id, data in gate_history.items():
        for layer_idx in range(len(data['attention_gates'])):
            att_gates = data['attention_gates'][layer_idx]
            # 커스텀 분석 로직
            pass
```

### 새로운 시각화 추가

```python
# GateAnalyzer 클래스 확장
class CustomGateAnalyzer(GateAnalyzer):
    def custom_visualization(self):
        # 새로운 시각화 로직
        pass
```

## 🚨 주의사항

1. **메모리 사용량**: Gate 추적은 추가 메모리를 사용하므로 큰 배치나 긴 시퀀스에서 주의
2. **성능 영향**: Gate 값 저장으로 인한 약간의 성능 저하 가능
3. **GPU 메모리**: Gate 값을 CPU로 이동하여 저장하므로 GPU 메모리 사용량은 크게 증가하지 않음

## 🔍 문제 해결

### 일반적인 문제

1. **Gate tracking이 활성화되지 않음**
   ```python
   # 해결: 명시적으로 활성화
   model.avigate_fusion.enable_gate_tracking()
   ```

2. **샘플 ID가 설정되지 않음**
   ```python
   # 해결: 추론 전에 샘플 ID 설정
   model.avigate_fusion.set_current_sample_id("sample_id")
   ```

3. **분석 결과가 생성되지 않음**
   ```python
   # 해결: gate_history 확인
   print(len(model.avigate_fusion.gate_history))
   ```

### 디버깅

```python
# Gate 추적 상태 확인
print(f"Gate tracking enabled: {model.avigate_fusion.track_gates}")
print(f"Number of tracked samples: {len(model.avigate_fusion.gate_history)}")
print(f"Current sample ID: {model.avigate_fusion.current_sample_id}")

# Gate 값 직접 확인
for sample_id, data in model.avigate_fusion.gate_history.items():
    print(f"Sample {sample_id}:")
    for layer_idx in range(len(data['attention_gates'])):
        att_gate = data['attention_gates'][layer_idx]
        ffn_gate = data['ffn_gates'][layer_idx]
        print(f"  Layer {layer_idx}: att_shape={att_gate.shape}, ffn_shape={ffn_gate.shape}")
```

## 📝 예시 워크플로우

### 1. 모델 성능 분석

```python
# 모델 로드 및 gate 추적 활성화
model.avigate_fusion.enable_gate_tracking()

# 검증 데이터로 추론
for batch in val_dataloader:
    model.avigate_fusion.set_current_sample_id(f"val_{batch['qid']}")
    outputs = model(batch)

# 분석 및 리포트 생성
analyzer = GateAnalyzer(model.avigate_fusion)
analyzer.analyze_all_samples("./validation_analysis")
```

### 2. 하이퍼파라미터 튜닝

```python
# 다양한 gating_type 비교
for gating_type in ['global', 'clipwise', 'elementwise']:
    model = create_model(gating_type=gating_type)
    # ... 추론 및 분석
    # 결과 비교하여 최적 설정 선택
```

### 3. 디버깅 및 모델 이해

```python
# 특정 실패 케이스 분석
failed_samples = ["sample_123", "sample_456"]
for sample_id in failed_samples:
    model.avigate_fusion.set_current_sample_id(sample_id)
    outputs = model(batch)
    
# 실패 케이스의 gate 패턴 분석
model.avigate_fusion.compare_samples_gates(failed_samples, "./debug_analysis")
```

이 시스템을 통해 AVIGATEFusionCustom 모듈의 gate 동작을 상세히 분석하고, 모델의 audio-video 융합 전략을 깊이 이해할 수 있습니다.
