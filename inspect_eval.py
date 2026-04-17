import json

with open("eval_condition_A.json") as f:
    d = json.load(f)

print(f"Total predictions: {d['num_samples']}")
print(f"Metrics: {d['metrics']}\n")

for i, p in enumerate(d["predictions"][:5]):
    print(f"=== Sample {i+1} ===")
    print(f"GT:   {p['gt']}")
    print(f"PRED: {p['pred']}")
    print(f"RAW:  {repr(p['raw'][:400])}")
    print()
