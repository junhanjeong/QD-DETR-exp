# MR 성능 측정용 스크립트
# Random baseline의 MR 성능을 측정하기 위해 제작
# 결과를 터미널에 출력하고 json 파일로 저장
# QD-DETR-Exp/ 경로에서 `PYTHONPATH=. python supplementary/eval_random_baseline.py` 로 실행
from standalone_eval.eval import eval_submission, load_jsonl
import os

# 경로 세팅
pred_file_name = "hl_random_val_pred_2018.jsonl"
gt_path = "../data/highlight_val_release.jsonl"

# 절대 경로로 변환
pred_file_name = os.path.join(os.path.dirname(__file__), pred_file_name)
gt_path = os.path.join(os.path.dirname(__file__), gt_path)

# 파일 로드
submission = load_jsonl(pred_file_name)
ground_truth = load_jsonl(gt_path)

# 평가 (MR만)
results = eval_submission(submission, ground_truth, verbose=True)

# 결과 출력
import json
print(json.dumps(results, indent=4))

# 결과 파일 이름 설정
result_file_name = pred_file_name.replace('.jsonl', '_metrics.json')

# 결과를 json 파일로 저장
with open(result_file_name, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)