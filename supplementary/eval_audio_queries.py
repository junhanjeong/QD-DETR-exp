import sys
import os
import json
from pathlib import Path

# 프로젝트 루트 경로를 시스템 경로에 추가합니다.
# 이 스크립트가 'supplementary' 폴더 안에 있다고 가정합니다.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from standalone_eval.eval import eval_submission, load_jsonl

def evaluate_audio_queries(prediction_path, ground_truth_path, output_path=None):
    """
    오디오 관련 쿼리에 대한 성능을 측정하고 결과를 저장합니다.

    Args:
        prediction_path (str): 모델 예측 결과 파일 경로 (e.g., best_hl_val_preds.jsonl)
        ground_truth_path (str): 오디오 관련 쿼리만 필터링된 ground truth 파일 경로 (e.g., highlight_val_release_audio.jsonl)
        output_path (str, optional): 결과 metrics를 저장할 JSON 파일 경로. 지정하지 않으면 예측 파일과 동일한 디렉토리에 생성됩니다.
    """
    print("평가를 시작합니다...")
    print(f"예측 파일: {prediction_path}")
    print(f"Ground Truth 파일: {ground_truth_path}")

    # 데이터 로드
    try:
        submission = load_jsonl(prediction_path)
        ground_truth = load_jsonl(ground_truth_path)
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다 - {e.filename}")
        return

    # Ground Truth에 있는 qid만 필터링
    gt_qids = {item['qid'] for item in ground_truth}
    filtered_submission = [item for item in submission if item['qid'] in gt_qids]

    print(f"전체 예측 {len(submission)}개 중, 오디오 관련 Ground Truth에 해당하는 {len(filtered_submission)}개에 대해 평가를 진행합니다.")

    if not filtered_submission:
        print("평가할 예측 데이터가 없습니다. QID가 일치하는지 확인해주세요.")
        return

    # 평가 수행
    results = eval_submission(filtered_submission, ground_truth, verbose=True)

    # 결과 파일 경로 설정
    if output_path is None:
        pred_path_obj = Path(prediction_path)
        output_filename = pred_path_obj.name.replace('.jsonl', '_audio_metrics.json')
        output_path = pred_path_obj.parent / output_filename

    # 결과 출력
    print("\n--- 평가 결과 ---")
    print(json.dumps(results, indent=4))
    print("-----------------\n")

    # 결과를 JSON 파일로 저장
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"성능 측정 결과를 다음 파일에 저장했습니다: {output_path}")
    except IOError as e:
        print(f"오류: 결과를 파일에 쓰는 데 실패했습니다 - {e}")


if __name__ == '__main__':
    # --- 사용 예시 ---
    # 1. 예측 결과 파일 경로를 지정합니다.
    # 예: results/hl-video_tef-audio_experiment-2025_07_04_12_02_18/best_hl_val_preds.jsonl
    # 이 경로는 QD-DETR-exp 폴더를 기준으로 합니다.
    pred_file = "results/hl-video_tef-temporal_gate-2025_07_13_20_39_17/best_hl_val_preds.jsonl"
    
    # 2. 오디오 관련 쿼리에 대한 Ground Truth 파일 경로를 지정합니다.
    # 예: data/highlight_val_release_audio.jsonl
    gt_file = "data/highlight_val_release_audio.jsonl"

    # 절대 경로로 변환
    abs_pred_path = project_root / pred_file
    abs_gt_path = project_root / gt_file
    
    evaluate_audio_queries(str(abs_pred_path), str(abs_gt_path))
