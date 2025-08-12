# QD-DETR을 Base로 Audio 논문 구현용 코드
## 폴더 디렉토리 구조
```bash
root
├── features
└── QD-DETR-Exp
    ├── README.md
    ├── data
    │   ├── LICENSE
    │   ├── README.md
    │   ├── highlight_test_release.jsonl
    │   ├── highlight_train_release.jsonl
    │   ├── highlight_val_release.jsonl
    │   ├── subs_train.jsonl
    │   └── tvsum
    ├── qd_detr
    ├── requirements.txt
    ├── results
    ├── standalone_eval
    ├── supplementary # 논문 실험용 코드 (보조, 분석용)
    └── utils
```
features와 data는 QD-DETR에서 다운받을 수 있습니다.
https://github.com/wjun0830/QD-DETR

## AVIGATE Custom Gating (Audio-Visual Fusion)

본 레포에는 AVIGATE 커스텀 게이팅이 포함되어 있습니다. 핵심 변경점은 다음과 같습니다.
- MHA 게이트: `sigmoid((raw + bias) / temp) * scale` 형태로 안정적인 0~1 게이팅 적용 (기본값: `temp=1.5`, `bias=1.0`, `scale=1.0`).
- FFN 게이트: `tanh(raw) * α` 형태의 잔차 보강(기본값: `α=0.3`).
- 선택적 FiLM: FFN에 채널별 scale(1+gate)과 shift(β) 적용을 플래그로 활성화 가능.
- Δ LayerNorm: MHA 및 FFN의 delta에 LayerNorm을 적용해 분포를 안정화.

### 새 Config 플래그
- `--mha_gate_temp`: Sigmoid MHA 게이트 온도(포화 감소, 기본 1.5)
- `--mha_gate_bias_init`: MHA 게이트 바이어스 초기값(기본 1.0)
- `--mha_gate_scale_init`: MHA 게이트 스케일(기본 1.0)
- `--ffn_gate_alpha_init`: FFN tanh 게이트 스케일 α(기본 0.3)
- `--ffn_gate_film`: FFN FiLM(scale+shift) 활성화 플래그(기본 비활성)
- `--ffn_gate_beta_init`: FFN FiLM shift β 스케일(기본 0.3)

게이트 로깅 옵션(추론 시)
- `--gate_log`: 게이트 값을 수집/저장
- `--gate_sample_count`: 로깅할 샘플 수(-1이면 전체)
- `--gate_save_path`: 저장 경로(미지정 시 results 디렉토리 내 기본값 사용)

### 학습 실행 스크립트
입력 경로/차원 인자는 사용 환경에 맞게 채워 넣어 주세요.

1) 기본(안정적) 게이팅
```
bash qd_detr/scripts/run_avigate_custom.sh \
  --dset_name tvsum \
  --train_path <train.jsonl> \
  --eval_path <val.jsonl> \
  --v_feat_dirs <vfeat_dir> \
  --t_feat_dir <tfeat_dir> \
  --a_feat_dir <afeat_dir> \
  --v_feat_dim 1024 --t_feat_dim 768 --a_feat_dim 512 \
  --results_root results --exp_id avigate_sigmoid_tanh
```

2) FFN FiLM 활성화(스케일+시프트)
```
bash qd_detr/scripts/run_avigate_custom_film.sh \
  --dset_name tvsum \
  --train_path <train.jsonl> \
  --eval_path <val.jsonl> \
  --v_feat_dirs <vfeat_dir> \
  --t_feat_dir <tfeat_dir> \
  --a_feat_dir <afeat_dir> \
  --v_feat_dim 1024 --t_feat_dim 768 --a_feat_dim 512 \
  --results_root results --exp_id avigate_film
```

3) MHA 게이트를 더 강하게(바이어스↑)
```
bash qd_detr/scripts/run_avigate_custom_mha_strong.sh \
  --dset_name tvsum \
  --train_path <train.jsonl> \
  --eval_path <val.jsonl> \
  --v_feat_dirs <vfeat_dir> \
  --t_feat_dir <tfeat_dir> \
  --a_feat_dir <afeat_dir> \
  --v_feat_dim 1024 --t_feat_dim 768 --a_feat_dim 512 \
  --results_root results --exp_id avigate_mha_strong
```

### 추론 실행 스크립트
체크포인트 경로(`--resume`)를 지정하여 사용하세요.

1) 기본 게이팅 추론
```
bash qd_detr/scripts/infer_avigate_custom.sh \
  --dset_name tvsum \
  --eval_path <val.jsonl> \
  --v_feat_dirs <vfeat_dir> \
  --t_feat_dir <tfeat_dir> \
  --a_feat_dir <afeat_dir> \
  --v_feat_dim 1024 --t_feat_dim 768 --a_feat_dim 512 \
  --results_root results --exp_id avigate_sigmoid_tanh \
  --resume <ckpt_path>
```

2) FiLM 활성화 추론(+게이트 로깅 예시)
```
bash qd_detr/scripts/infer_avigate_custom_film.sh \
  --dset_name tvsum \
  --eval_path <val.jsonl> \
  --v_feat_dirs <vfeat_dir> \
  --t_feat_dir <tfeat_dir> \
  --a_feat_dir <afeat_dir> \
  --v_feat_dim 1024 --t_feat_dim 768 --a_feat_dim 512 \
  --results_root results --exp_id avigate_film \
  --resume <ckpt_path> \
  --gate_log --gate_sample_count 50
```

권장 팁
- MHA 게이트가 0 근처로 수렴하는 경우 `--mha_gate_bias_init`를 1.5~2.0으로 올려 정보 흐름을 보장하세요.
- FFN 기여가 과하면 `--ffn_gate_alpha_init`를 0.2~0.3 범위로 스윕하여 균형을 맞추세요.

## 환경 설치
```bash
# conda 환경
conda create -n env_name python=3.7

# ffmpeg 설치
conda install ffmpeg

# av 설치
conda install -c conda-forge av==8.0.3

# 만약 conda 대신 apt로 할거면
sudo apt update
sudo apt install -y ffmpeg libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libswscale-dev libavutil-dev
sudo apt install -y build-essential pkg-config python3-dev
pip install av==8.0.3

# pytorch 설치
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# requirements.txt 설치
pip install -r requirements.txt

```

## 추론 시 Gate 저장 및 시각화

inference를 할 때 gate를 저장하려면 아래와 같이 실행합니다.

```bash
bash qd_detr/scripts/inference.sh \
  results/hl-video_tef-avigate_custom_elementwise_0.5_2018_l4_h4-2025_07_18_16_07_40/model_best.ckpt \
  val \
  --gate_log
```

저장된 gate 로그를 시각화하려면 다음 명령을 사용합니다.

```bash
python supplementary/visualize_gate_logs.py \
  --input results/hl-video_tef-avigate_custom_elementwise_0.5_2018_l4_h4-2025_07_18_16_07_40/gate_logs_val_Nall.jsonl \
  --num-samples 50
```
