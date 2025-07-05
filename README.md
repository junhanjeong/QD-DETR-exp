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