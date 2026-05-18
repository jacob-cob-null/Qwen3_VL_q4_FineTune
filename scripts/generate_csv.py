"""
Generate validated_benchmark.csv from raw eval predictions.
Standardizes all models to the same 9-field schema and stratified scoring.
"""
import json, os, sys, csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import compute_metrics

CORE_TRIAD    = ["date", "total_amount", "patient_name"]
ANCHOR_FIELDS = ["date", "total_amount"]

def filt(record, fields):
    def _f(d): return {k: v for k, v in (d or {}).items() if k in fields}
    return {**record, "prediction": _f(record.get("prediction")), "ground_truth": _f(record.get("ground_truth"))}

def norm(p):
    return {
        "prediction":    p.get("pred", p.get("prediction", {})),
        "ground_truth":  p.get("gt",   p.get("ground_truth", {})),
        "source_dataset": p.get("source", p.get("source_dataset", "?")),
        "tier":          p.get("tier"),
        "stress_level":  p.get("stress_level"),
    }

def score_file(path):
    if not os.path.exists(path): return None
    d = json.load(open(path))
    recs = [norm(p) for p in d.get("predictions", [])]
    
    donut = [r for r in recs if "donut" in r["source_dataset"] or "invoices" in r["source_dataset"]]
    synth = [r for r in recs if r.get("stress_level") == "high" or r.get("tier") == "Specialized"]
    sroie = [r for r in recs if "sroie" in r["source_dataset"]]
    
    ma = compute_metrics(donut) if donut else {}
    mb = compute_metrics([filt(r, CORE_TRIAD) for r in synth]) if synth else {}
    mb_full = compute_metrics(synth) if synth else {}
    ms = compute_metrics([filt(r, ANCHOR_FIELDS) for r in sroie]) if sroie else {}
    
    pf_a = ma.get("per_field", {})
    pf_b = mb.get("per_field", {})
    pf_b_full = mb_full.get("per_field", {})
    
    def f1(pf, k): return pf.get(k, {}).get("f1", 0.0)
    
    return {
        "a_f1": ma.get("macro_f1", 0), "a_em": ma.get("fuzzy_exact_match", 0),
        "a_date": f1(pf_a, "date"), "a_total": f1(pf_a, "total_amount"), "a_patient": f1(pf_a, "patient_name"),
        "a_balance": f1(pf_a, "balance_due"), "a_phic": f1(pf_a, "philhealth_number"),
        "a_diag": f1(pf_a, "diagnosis_code"), "a_proc": f1(pf_a, "procedure_code"),
        "a_origin": f1(pf_a, "service_origin"), "a_benefit": f1(pf_a, "philhealth_benefit"),
        "b_f1": mb.get("macro_f1", 0), "b_em": mb.get("fuzzy_exact_match", 0),
        "b_date": f1(pf_b, "date"), "b_total": f1(pf_b, "total_amount"), "b_patient": f1(pf_b, "patient_name"),
        "b_balance": f1(pf_b_full, "balance_due"), "b_phic": f1(pf_b_full, "philhealth_number"),
        "b_diag": f1(pf_b_full, "diagnosis_code"), "b_proc": f1(pf_b_full, "procedure_code"),
        "b_origin": f1(pf_b_full, "service_origin"), "b_benefit": f1(pf_b_full, "philhealth_benefit"),
        "s_f1": ms.get("macro_f1", 0), "s_em": ms.get("fuzzy_exact_match", 0)
    }

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "Zero-Shot":       os.path.join(BASE, "eval_condition_FineTune_zero_shot.json"),
    "Logic E1.5":      os.path.join(BASE, "clean_result_raw", "eval_condition_clean_data_clean_data-epoch-1.5.json"),
    "Composed 97:3":   os.path.join(BASE, "clean_result_dual", "eval_condition_L9_N0.json"),
}

results = {label: score_file(path) for label, path in RUNS.items()}
cols = list(RUNS.keys())

csv_path = os.path.join(BASE, "validated_benchmark.csv")
rows = [
    ["Tier A: Macro F1", "a_f1"], ["Tier A: Fuzzy EM", "a_em"],
    ["Tier A: date", "a_date"], ["Tier A: total_amount", "a_total"], ["Tier A: patient_name", "a_patient"],
    ["Tier A: balance_due", "a_balance"], ["Tier A: philhealth_number", "a_phic"],
    ["Tier A: diagnosis_code", "a_diag"], ["Tier A: procedure_code", "a_proc"],
    ["Tier A: service_origin", "a_origin"], ["Tier A: philhealth_benefit", "a_benefit"],
    ["Tier B: Macro F1 (Core Triad)", "b_f1"], ["Tier B: Fuzzy EM", "b_em"],
    ["Tier B: date", "b_date"], ["Tier B: total_amount", "b_total"], ["Tier B: patient_name", "b_patient"],
    ["Tier B: balance_due (all-field)", "b_balance"], ["Tier B: philhealth_number (all-field)", "b_phic"],
    ["Tier B: diagnosis_code (all-field)", "b_diag"], ["Tier B: procedure_code (all-field)", "b_proc"],
    ["Tier B: service_origin (all-field)", "b_origin"], ["Tier B: philhealth_benefit (all-field)", "b_benefit"],
    ["Anchor: SROIE Macro F1", "s_f1"], ["Anchor: SROIE Fuzzy EM", "s_em"]
]

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Metric"] + cols)
    for label, key in rows:
        writer.writerow([label] + [f"{results[c][key]:.4f}" if results[c] else "N/A" for c in cols])

print(f"CSV generated at: {csv_path}")
