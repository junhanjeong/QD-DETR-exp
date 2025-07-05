# QVHighlights val set에서 랜덤으로 moment 10개를 예측한 결과를 생성하는 스크립트 (video, audio 정보 없이)
# QVHighlgihts train set의 평균 window 길이(25.0837)과 유사한 25초 길이의 window 10개를 랜덤으로 생성합니다.
# seed를 설정할 수 있습니다.

import json
import random

def generate_predictions(input_path='hl_val_query_sample.jsonl', output_path='hl_random_val_pred.jsonl', seed=2018):
    random.seed(seed)  # seed 고정
    output_path = output_path.replace('.jsonl', f'_{seed}.jsonl')

    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line.strip())
            # 0~125 중 서로 다른 10개
            starts = random.sample(range(0, 126), 10)
            windows = [[start, start + 25, 0.5] for start in starts]
            
            output_entry = {
                'qid': data['qid'],
                'query': data['query'],
                'vid': data['vid'],
                'pred_relevant_windows': windows
            }
            fout.write(json.dumps(output_entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    generate_predictions(seed=2018)