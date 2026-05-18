"""
Stratified IEEE Success Matrix — Source-Aware Schema Filtering.

Implements three evaluation masks:
  Full Schema  (9 fields) -> Donut clean invoices     [Tier A Specialist]
  Core Triad   (3 fields) -> High-noise paige_synth   [Tier B Operational]
  Anchor Only  (2 fields) -> SROIE general receipts   [Anchor - informational]

Run from the FineTune root:
    .\.venv311\Scripts\python.exe scripts\rescore_stratified.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import compute_metrics, _prune_to_core, CORE_FIELDS

# ---------------------------------------------------------------------------
# Schema Masks
# ---------------------------------------------------------------------------
FULL_SCHEMA   = [
    "date", "total_amount", "balance_due",
    "patient_name", "philhealth_number",
    "diagnosis_code", "procedure_code",
    "service_origin", "philhealth_benefit",
]
CORE_TRIAD    = ["date", "total_amount", "patient_name"]
ANCHOR_FIELDS = ["date", "total_amount"]

def filter_fields(record, allowed_fields):
    """Return a copy of a result record with GT and prediction pruned to allowed_fields."""
    def _filter(d):
        return {k: v for k, v in (d or {}).items() if k in allowed_fields}
    return {
        **record,
        "prediction":   _filter(record.get("prediction")),
        "ground_truth": _filter(record.get("ground_truth")),
    }

# ---------------------------------------------------------------------------
# Load saved predictions
# ---------------------------------------------------------------------------
_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# compose_adapters.py saves to root; clean_result_dual is the archive location
for _candidate in [
    os.path.join(_base, "eval_condition_L9_N0.json"),
    os.path.join(_base, "clean_result_dual", "eval_condition_L9_N0.json"),
]:
    if os.path.exists(_candidate):
        EVAL_FILE = _candidate
        break
else:
    import glob
    candidates = sorted(
        glob.glob(os.path.join(_base, "**", "eval_condition_L*.json"), recursive=True),
        key=os.path.getmtime, reverse=True
    )
    if candidates:
        EVAL_FILE = candidates[0]
        print(f"  Note: Using fallback eval file: {os.path.basename(EVAL_FILE)}")
    else:
        print("ERROR: No eval_condition_L*.json found.")
        sys.exit(1)

print(f"  Scoring: {os.path.basename(EVAL_FILE)}")
d = json.load(open(EVAL_FILE))
preds = d.get("predictions", [])

def normalize_record(p):
    return {
        "prediction":    p.get("pred", p.get("prediction", {})),
        "ground_truth":  p.get("gt",   p.get("ground_truth", {})),
        "source_dataset": p.get("source", p.get("source_dataset", "unknown")),
        "tier":          p.get("tier"),
        "stress_level":  p.get("stress_level"),
    }

records = [normalize_record(p) for p in preds]

# ---------------------------------------------------------------------------
# Bucket by source + noise level
# ---------------------------------------------------------------------------
tier_a_specialist = []   # Donut / clean medical invoices
tier_b_operational = []  # High-noise pAIge synthetic
anchor_sroie = []        # SROIE general receipts (informational)

for r in records:
    src = r.get("source_dataset", "")
    sl  = r.get("stress_level", "")

    if "sroie" in src.lower():
        anchor_sroie.append(filter_fields(r, ANCHOR_FIELDS))
    elif sl == "high" or r.get("tier") == "Specialized":
        tier_b_operational.append(filter_fields(r, CORE_TRIAD))
    else:
        tier_a_specialist.append(r)   # Full Schema — no pruning

# ---------------------------------------------------------------------------
# Compute metrics per bucket
# ---------------------------------------------------------------------------
m_a    = compute_metrics(tier_a_specialist)  if tier_a_specialist  else None
m_b    = compute_metrics(tier_b_operational) if tier_b_operational else None
m_anch = compute_metrics(anchor_sroie)       if anchor_sroie        else None

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
SEP  = "=" * 68
SEP2 = "-" * 68

print(SEP)
print("  IEEE Stratified Success Matrix")
print(f"  Source: {EVAL_FILE}  (97:3 Composed, 1024px)")
print(SEP)

# --- Tier A Specialist ---
if m_a:
    gate = "PASS" if m_a["macro_f1"] >= 0.94 else "FAIL"
    em_gate = "PASS" if m_a["fuzzy_exact_match"] >= 0.90 else "---"
    print(f"\n[Tier A] Specialist Gate  -- Clean Medical Invoices (n={len(tier_a_specialist)})")
    print(f"  Schema : Full ({len(FULL_SCHEMA)} fields)")
    print(f"  Macro F1  : {m_a['macro_f1']:.4f}    Gate >= 0.94   [{gate}]")
    print(f"  Fuzzy  EM : {m_a['fuzzy_exact_match']:.4f}    Gate >= 0.90   [{em_gate}]")
    pf = m_a.get("per_field", {})
    for f in ["date", "total_amount", "patient_name"]:
        if f in pf:
            e = pf[f]
            print(f"    {f:20s}  F1={e['f1']:.4f}  P={e['precision']:.4f}  R={e['recall']:.4f}")

# --- Tier B Operational ---
if m_b:
    gate = "PASS" if m_b["macro_f1"] >= 0.75 else "FAIL"
    em_gate = "PASS" if m_b["fuzzy_exact_match"] >= 0.30 else "FAIL"
    print(f"\n[Tier B] Operational Gate -- High-Noise Synthetic (n={len(tier_b_operational)})")
    print(f"  Schema : Core Triad {CORE_TRIAD}")
    print(f"  Macro F1  : {m_b['macro_f1']:.4f}    Gate >= 0.75   [{gate}]")
    print(f"  Fuzzy  EM : {m_b['fuzzy_exact_match']:.4f}    Gate >= 0.30   [{em_gate}]")
    pf = m_b.get("per_field", {})
    for f in CORE_TRIAD:
        if f in pf:
            e = pf[f]
            print(f"    {f:20s}  F1={e['f1']:.4f}  P={e['precision']:.4f}  R={e['recall']:.4f}")

# --- Anchor SROIE (informational) ---
if m_anch:
    print(f"\n[Anchor] SROIE General Receipts -- Informational (n={len(anchor_sroie)})")
    print(f"  Schema : Anchor {ANCHOR_FIELDS}")
    print(f"  Macro F1  : {m_anch['macro_f1']:.4f}    (no gate -- cross-domain anchor only)")
    print(f"  Fuzzy  EM : {m_anch['fuzzy_exact_match']:.4f}")

# --- Summary table ---
print(f"\n{SEP}")
print("  SUMMARY TABLE")
print(SEP2)
print(f"  {'Category':<28} {'n':>4}  {'Schema':<12}  {'F1':>6}  {'EM':>6}  {'Gate'}")
print(SEP2)
if m_a:
    g = "PASS" if m_a["macro_f1"] >= 0.94 else "FAIL"
    print(f"  {'Tier A  Specialist (Clean)':<28} {len(tier_a_specialist):>4}  {'Full (9)':<12}  {m_a['macro_f1']:>6.4f}  {m_a['fuzzy_exact_match']:>6.4f}  {g}")
if m_b:
    g = "PASS" if m_b["macro_f1"] >= 0.75 else "FAIL"
    print(f"  {'Tier B  Operational (Stress)':<28} {len(tier_b_operational):>4}  {'Core (3)':<12}  {m_b['macro_f1']:>6.4f}  {m_b['fuzzy_exact_match']:>6.4f}  {g}")
if m_anch:
    print(f"  {'Anchor  SROIE (informational)':<28} {len(anchor_sroie):>4}  {'Anchor (2)':<12}  {m_anch['macro_f1']:>6.4f}  {m_anch['fuzzy_exact_match']:>6.4f}  N/A")
print(SEP)
