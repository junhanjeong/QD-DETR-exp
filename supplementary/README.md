# 논문 실험용 보조 코드 모음 (분석용)
```bash
# supplementary/ 폴더에서 실행
python calc_mean_window_length_hl_train.py # QVHighlights 평균 window 길이 구하는 코드 (train set) # 평균길이: 20.87
python make_hl_random_baseline.py # video, audio 정보 없이 랜덤으로 결과 예측하는 코드 (seed 설정 가능)

# ..에서 실행
PYTHONPATH=. python supplementary/eval_random_baseline.py # 랜덤 baseline 성능 측정 (결과 출력 및 저장)
```