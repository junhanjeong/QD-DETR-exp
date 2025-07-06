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