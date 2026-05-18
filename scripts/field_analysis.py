"""Analyze per-field accuracy on the composed eval results."""
import json
from collections import Counter

d = json.load(open("eval_condition_L9_N0.json"))
preds = d.get("predictions", [])

# Filter to paige_synthetic only
synth = [p for p in preds if p.get("tier") == "Specialized" or "paige" in str(p.get("source",""))]

# If no source field, try stress_level
if not synth:
    synth = [p for p in preds if p.get("stress_level") == "high"]

print(f"Found {len(synth)} synthetic/specialized samples out of {len(preds)} total")
if not synth:
    # Show what we have
    for p in preds[:3]:
        print(json.dumps(p, indent=2)[:500])
    exit()

field_correct = Counter()
field_total = Counter()
field_errors = {}
em_pass = 0

for s in synth:
    pred = s.get("pred", {})
    gt = s.get("gt", {})
    if isinstance(pred, str):
        try: pred = json.loads(pred)
        except: pred = {}
    if isinstance(gt, str):
        try: gt = json.loads(gt)
        except: gt = {}

    all_correct = True
    for key in gt:
        field_total[key] += 1
        pred_val = str(pred.get(key, "")).strip()
        gt_val = str(gt[key]).strip()
        if pred_val.lower() == gt_val.lower():
            field_correct[key] += 1
        else:
            all_correct = False
            if key not in field_errors:
                field_errors[key] = []
            if len(field_errors[key]) < 3:
                field_errors[key].append({"gt": gt_val, "pred": pred_val})
    if all_correct:
        em_pass += 1

print(f"\n=== Per-Field Accuracy on Strong-Noise Synthetic (n={len(synth)}) ===")
for k in sorted(field_total.keys()):
    acc = field_correct[k] / field_total[k] * 100
    print(f"  {k:25s}  {field_correct[k]:3d}/{field_total[k]:3d}  ({acc:.1f}%)")

print(f"\n  Exact Match: {em_pass}/{len(synth)} ({em_pass/len(synth)*100:.1f}%)")

print("\n=== Sample Errors (up to 3 per field) ===")
for k in sorted(field_errors.keys()):
    print(f"  [{k}]")
    for e in field_errors[k]:
        gt_val = e["gt"]
        pred_val = e["pred"]
        print(f"    GT:   {gt_val}")
        print(f"    PRED: {pred_val}")
        print()
