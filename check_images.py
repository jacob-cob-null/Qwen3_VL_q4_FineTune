import json, os, random

random.seed(0)
lines = [json.loads(l) for l in open("Datasets/Training_Data/golden/merged/all_sources_train.jsonl") if l.strip()]
sample = random.sample(lines, min(32, len(lines)))
paths = [s.get("image_path", "") for s in sample]
exists = [os.path.exists(p) for p in paths]
print(f"32 sampled: {sum(exists)} have existing images, {len(exists)-sum(exists)} missing")
print("First 5 paths:")
for p, e in zip(paths[:5], exists[:5]):
    status = "OK" if e else "MISSING"
    print(f"  [{status}] {p}")
