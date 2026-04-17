import json, os

lines = [json.loads(l) for l in open("Datasets/Testing_Data/sroie_2019_v2/canonical/test.jsonl") if l.strip()][:20]
paths = [s.get("image_path", "") for s in lines]
exists = [os.path.exists(p) for p in paths]
print(f"20 test samples: {sum(exists)} have existing images, {len(exists)-sum(exists)} missing")
for p, e in zip(paths[:5], exists[:5]):
    status = "OK" if e else "MISSING"
    print(f"  [{status}] {p}")
