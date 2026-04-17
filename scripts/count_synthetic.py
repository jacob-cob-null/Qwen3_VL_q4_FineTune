from pathlib import Path
p=Path("Datasets/Training_Data/golden/merged/all_sources_train.jsonl")
if not p.exists():
    raise SystemExit(f"File not found: {p}")
t=p.read_text(encoding="utf-8")
print(t.count('"is_synthetic": true'))
print(t.count('"is_synthetic": false'))
