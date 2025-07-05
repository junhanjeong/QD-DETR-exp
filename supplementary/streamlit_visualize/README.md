# 총 3가지 버전의 Streamlit Demo
1. QVhighlights train/val 확인용 데모
2. 모델이 QVHighlights val에서 잘 예측한 Query 순으로 정렬한 데모
3. 모델 2개의 성능 비교하는 데모

## 설치 및 실행
기존 환경이 아닌 새로운 환경을 만들어서, 그 환경에 requirements.txt를 설치해주세요.

### 1. 의존성 설치
```bash
pip install -r requirements_streamlit.txt
```

### 2. 애플리케이션 실행
```bash
# 1번
streamlit run hl_dataset_viewer.py

# 2번
streamlit run model_pred_sorted_by_AP.py

# 3번
streamlit run model_comparison_viewer.py
```