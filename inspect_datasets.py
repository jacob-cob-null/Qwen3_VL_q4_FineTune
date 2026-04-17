"""
Deep-dive: inspect what label fields look like in cord_v2 and invoices_donut_v1
to understand why label fill is so low, and whether the chat messages
in paige_synthetic are actually valid.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MERGED = "Datasets/Training_Data/golden/merged/all_sources_train.jsonl"
FIELDS = ["date","patient_name","philhealth_number","diagnosis_code",
          "procedure_code","total_amount","philhealth_benefit","balance_due"]

samples = {"cord_v2": [], "invoices_donut_v1": [], "paige_synthetic": []}

with open(MERGED, encoding="utf-8") as f:
    for raw in f:
        try:
            item = json.loads(raw)
        except Exception:
            continue
        src = item.get("source_dataset", "")
        if src in samples and len(samples[src]) < 2:
            samples[src].append(item)

for src, items in samples.items():
    print(f"\n{'='*60}")
    print(f"  SOURCE: {src}")
    print(f"{'='*60}")
    for i, item in enumerate(items):
        print(f"\n  Sample {i+1}:")
        print(f"  ground_truth: {item.get('ground_truth', {})}")
        msgs = item.get("messages", [])
        print(f"  messages ({len(msgs)} turns):")
        for m in msgs:
            role = m.get("role","?") if isinstance(m, dict) else "raw"
            content = m.get("content","") if isinstance(m, dict) else str(m)
            content_str = str(content)[:150]
            print(f"    [{role}] {content_str}")
