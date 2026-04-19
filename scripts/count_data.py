import json
import os

TARGET = [
    "date", "patient_name", "philhealth_number", "diagnosis_code",
    "procedure_code", "total_amount", "philhealth_benefit", "balance_due",
]

path = "Datasets/Training_Data/golden/merged/all_sources_train.jsonl"
real = 0
synthetic = 0
bad = 0
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                it = json.loads(line)
            except Exception:
                bad += 1
                continue
            gt = it.get("ground_truth", {})
            if not any(gt.get(f) is not None for f in TARGET):
                continue
            if it.get("is_synthetic", False):
                synthetic += 1
            else:
                real += 1
    print(f"real:{real} synthetic:{synthetic} skipped_invalid_json:{bad}")
else:
    print(f"merged file missing: {path}")
