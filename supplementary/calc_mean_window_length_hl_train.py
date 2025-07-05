# QVHighlgihts train set의 평균 window 길이를 구하는 코드

import json

lengths = []

with open('../data/highlight_train_release.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        for window in item.get("relevant_windows", []):
            if isinstance(window, list) and len(window) == 2:
                start, end = window
                lengths.append(end - start)

if lengths:
    avg_length = sum(lengths) / len(lengths)
    print(f"전체 relevant_windows의 평균 길이: {avg_length:.4f}")
else:
    print("relevant_windows가 없습니다.")
