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
