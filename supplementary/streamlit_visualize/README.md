## 주의
기존 환경이 아닌 새로운 환경을 만들어서, 그 환경에 requirements.txt를 설치해주세요.

## 3가지 시각화 데모
아래 3가지 streamlit demo 코드를 작성했습니다.
### QVHighlights train/val 시각화
- 쿼리 번역, Youtube 영상, moment 구간 제공
- 쿼리 검색 가능
### hl_val_pred.jsonl을 바탕으로 AP/R1이 높은 Query부터 볼 수 있도록 시각화
- 모델이 잘 맞추는 Query 탐색 가능
- AP@Avg, AP@0.5, R1@0.5 등의 지표를 선택하여 정렬 가능
### hl_val_pred1.jsonl, hl_val_pred2.jsonl를 바탕으로 2개의 모델이 예측한 결과를 비교하는 시각화
- 2개의 모델 비교 시각화
- ex. video-only 모델과 video+audio 모델 결과물을 분석해서 video-only가 video+audio보다 잘 맞추는 쿼리 (AP@AVG가 더 높은 쿼리) 등을 알아낼 수 있음