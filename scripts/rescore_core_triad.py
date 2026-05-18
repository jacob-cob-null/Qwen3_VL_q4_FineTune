"""
Rescore eval_condition_L9_N0.json using Context-Aware Pruning.
Shows before/after metrics without re-running inference.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import compute_metrics, _prune_to_core, CORE_FIELDS, REDACTED_UNDER_NOISE

d = json.load(open("eval_condition_L9_N0.json"))
preds = d.get("predictions", [])

# Split by tier
high_noise = [p for p in preds if p.get("stress_level") == "high"]
clean_docs  = [p for p in preds if p.get("stress_level") != "high"]

# Normalize keys: saved JSON uses 'gt'/'pred', compute_metrics expects 'ground_truth'/'prediction'
def normalize_record(p):
    return {
        "prediction":    p.get("pred", p.get("prediction", {})),
        "ground_truth":  p.get("gt",   p.get("ground_truth", {})),
        "source_dataset": p.get("source", p.get("source_dataset", "unknown")),
        "tier":          p.get("tier"),
        "stress_level":  p.get("stress_level"),
    }

high_noise_norm = [normalize_record(p) for p in high_noise]
clean_norm      = [normalize_record(p) for p in clean_docs]

# Score high-noise samples: BEFORE (full schema) vs AFTER (Core Triad only)
high_before = compute_metrics(high_noise_norm)
high_after  = compute_metrics([_prune_to_core(r) for r in high_noise_norm])

# Score clean samples (always full schema)
clean_full = compute_metrics(clean_norm)

SEP = "=" * 65
print(SEP)
print("IEEE Success Matrix - Context-Aware Pruning Re-Score")
print("  Source: eval_condition_L9_N0.json (97:3 Composed, 1024px)")
print(SEP)

gate_a = "PASS" if clean_full["macro_f1"] >= 0.94 else "FAIL"
print(f"\nTier A - Integrity Gate (Clean Documents, n={len(clean_docs)})")
print(f"  Fields: Full Schema (all 9 fields)")
print(f"  Macro F1:  {clean_full['macro_f1']:.4f}")
print(f"  Fuzzy EM:  {clean_full['fuzzy_exact_match']:.4f}")
print(f"  Gate:      {gate_a}  (target >= 0.94)")

gate_b = "PASS" if high_after["macro_f1"] >= 0.75 else "FAIL"
print(f"\nTier B - Operational Gate (High-Noise, n={len(high_noise)})")
print(f"  Fields evaluated: {CORE_FIELDS}")
print(f"  Redacted:         {REDACTED_UNDER_NOISE}")
print(f"  BEFORE pruning:")
print(f"    Macro F1: {high_before['macro_f1']:.4f}   Fuzzy EM: {high_before['fuzzy_exact_match']:.4f}")
print(f"  AFTER pruning (Core Triad only):")
print(f"    Macro F1: {high_after['macro_f1']:.4f}   Fuzzy EM: {high_after['fuzzy_exact_match']:.4f}")
print(f"  Gate:      {gate_b}  (target >= 0.75)")

print(f"\nPer-Field Accuracy - Core Triad on High-Noise Samples:")
pf = high_after.get("per_field", {})
for f in CORE_FIELDS:
    if f in pf:
        entry = pf[f]
        print(f"  {f:15s}  F1={entry['f1']:.4f}  P={entry['precision']:.4f}  R={entry['recall']:.4f}")
