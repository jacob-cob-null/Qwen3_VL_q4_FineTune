import os
import re
import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # try jsonl: load first object or wrap as dict
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
                if len(lines) == 1:
                    return json.loads(lines[0])
                return {'__jsonl_count': len(lines)}
        except Exception:
            return None

def extract_eval_row(data, source_file):
    if not isinstance(data, dict):
        return None
    row = {
        'source_file': str(source_file.relative_to(ROOT)),
        'type': 'eval_condition',
        'condition_id': data.get('condition_id'),
        'adapter_path': data.get('adapter_path') or data.get('adapter'),
        'num_samples': data.get('num_samples'),
        'exact_match': None,
        'fuzzy_exact_match': None,
        'macro_f1': None,
        'balance_due_imputed': None,
        'date_anchor_errors': None,
    }
    metrics = data.get('metrics', {})
    if isinstance(metrics, dict):
        row['exact_match'] = metrics.get('exact_match')
        row['fuzzy_exact_match'] = metrics.get('fuzzy_exact_match')
        row['macro_f1'] = metrics.get('macro_f1')
    post = data.get('postprocess_summary', {})
    if isinstance(post, dict):
        row['balance_due_imputed'] = post.get('balance_due_imputed')
        row['date_anchor_errors'] = post.get('date_anchor_errors')

    # tiered metrics (metrics.tier_summary)
    tier_summary = metrics.get('tier_summary') if isinstance(metrics, dict) else None
    if isinstance(tier_summary, dict):
        for tier_name, tier_vals in tier_summary.items():
            # normalize tier_name to safe key
            safe = tier_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
            if isinstance(tier_vals, dict):
                for m_k, m_v in tier_vals.items():
                    key = f'tier_{safe}_{m_k}'
                    row[key] = m_v

    # per-domain summary if present
    per_domain = data.get('per_domain')
    if isinstance(per_domain, dict):
        for domain_name, domain_vals in per_domain.items():
            safe = domain_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')
            if isinstance(domain_vals, dict):
                # try to extract macro_f1 / fuzzy_em / other metrics if present
                for m_k, m_v in domain_vals.items():
                    # often nested structures; only grab scalars
                    if isinstance(m_v, (int, float, str)):
                        key = f'domain_{safe}_{m_k}'
                        row[key] = m_v

    # active fields summary
    af = metrics.get('active_fields') if isinstance(metrics, dict) else None
    if isinstance(af, list):
        row['active_field_count'] = len(af)
        row['active_fields_sample'] = ';'.join(map(str, af[:10]))
    return row

def extract_compose_rows(data, source_file):
    rows = []
    base = {
        'source_file': str(source_file.relative_to(ROOT)),
        'type': 'compose',
        'logic_adapter': data.get('logic_adapter'),
        'noise_adapter': data.get('noise_adapter'),
    }
    # best
    best = data.get('best')
    if isinstance(best, dict):
        r = base.copy()
        r.update({
            'label': best.get('label'),
            'logic_weight': best.get('logic_weight'),
            'noise_weight': best.get('noise_weight'),
            'structural_f1': best.get('structural_f1'),
            'specialized_em': best.get('specialized_em'),
            'passes_gate': best.get('passes_gate'),
        })
        rows.append(r)
    # sweep entries
    for s in data.get('sweep', []) or []:
        r = base.copy()
        r.update({
            'label': s.get('label'),
            'logic_weight': s.get('logic_weight'),
            'noise_weight': s.get('noise_weight'),
            'structural_f1': s.get('structural_f1'),
            'specialized_em': s.get('specialized_em'),
            'passes_gate': s.get('passes_gate'),
        })
        rows.append(r)
    return rows


def extract_adapter_folder_rows(root_final_res):
    rows = []
    root = Path(root_final_res)
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # look for epoch in the folder name
        m = re.search(r'epoch[-_]?(\d+\.?\d*)', name)
        epoch = m.group(1) if m else None
        # default ratio note: these are 70/30 adapters per user
        row = {
            'source_file': str(d.relative_to(ROOT)),
            'type': 'adapter_folder',
            'folder_name': name,
            'epoch': epoch,
            'train_ratio': '70/30',
            'note': 'Initial 70/30 setup; later revised to 2:1 for task focus',
        }
        rows.append(row)
    return rows

def main():
    candidates = []
    for root, dirs, files in os.walk(ROOT):
        for fn in files:
            if fn.endswith('.json') and ('eval_condition' in fn or 'compose_results' in fn):
                candidates.append(Path(root) / fn)

    rows = []
    for p in sorted(candidates):
        data = load_json(p)
        if data is None:
            continue
        if 'compose_results' in p.name:
            rows.extend(extract_compose_rows(data, p))
        else:
            r = extract_eval_row(data, p)
            if r:
                rows.append(r)

    # also include adapter folders from Eval_result/final_res (user-provided)
    final_res_dir = ROOT / 'Eval_result' / 'final_res'
    rows.extend(extract_adapter_folder_rows(final_res_dir))

    # classify rows into groups
    def classify(r):
        # composed rows
        if r.get('type') == 'compose':
            return 'composed'
        # adapter_folder rows were added as 70/30 by default
        if r.get('type') == 'adapter_folder' or r.get('train_ratio') == '70/30':
            return '70/30'
        # 2:1 approach heuristic: clean_data adapters or files under clean_result_raw
        ap = (r.get('adapter_path') or '') or ''
        sf = (r.get('source_file') or '')
        if 'clean_data' in ap or 'clean_result_raw' in sf or 'clean_data' in sf:
            return '2:1'
        # 2nd adapter eval heuristic: dual/combined evals or files under clean_result_dual
        if 'clean_result_dual' in sf or ('logic_adapter' in r and r.get('noise_adapter')):
            return '2nd_adapter_eval'
        return 'other'

    for r in rows:
        r['group'] = classify(r)

    # filter out any rows that are 70/30 or involve a noise adapter
    def is_noise_row(r):
        # check adapter_path, noise_adapter, logic_adapter
        for key in ('adapter_path', 'noise_adapter', 'logic_adapter', 'folder_name'):
            v = r.get(key) or ''
            if isinstance(v, str) and 'noise' in v.lower():
                return True
        if r.get('train_ratio') == '70/30':
            return True
        if r.get('group') == '70/30':
            return True
        return False

    filtered = [r for r in rows if not is_noise_row(r)]

    # order groups: 2:1, 2nd_adapter_eval, composed, other
    order = ['2:1', '2nd_adapter_eval', 'composed', 'other']
    ordered = []
    for g in order:
        ordered.extend([r for r in filtered if r.get('group') == g])

    # determine all columns
    fieldnames = []
    for r in ordered:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    out = ROOT / 'master_results.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ordered:
            writer.writerow(r)

    # write excluded rows (70/30 and noise adapters) to a separate CSV
    excluded = [r for r in rows if is_noise_row(r)]
    if excluded:
        # determine columns for excluded rows
        excl_fields = []
        for r in excluded:
            for k in r.keys():
                if k not in excl_fields:
                    excl_fields.append(k)
        out_excl = ROOT / 'master_results_excluded_70_30_noise.csv'
        with open(out_excl, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=excl_fields)
            writer.writeheader()
            for r in excluded:
                writer.writerow(r)
        print(f'Wrote {out} ({len(ordered)} rows, filtered {len(rows)-len(ordered)} rows)')
        print(f'Wrote excluded rows to {out_excl} ({len(excluded)} rows)')
    else:
        print(f'Wrote {out} ({len(ordered)} rows, filtered {len(rows)-len(ordered)} rows)')

if __name__ == '__main__':
    main()
"""
Validated compile: read raw predictions from all 3 eval JSONs and score them
identically through the SAME compute_metrics (9-field, with stratified pruning).
This eliminates the apples-to-oranges problem of different active_fields.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import compute_metrics

CORE_TRIAD    = ["date", "total_amount", "patient_name"]
ANCHOR_FIELDS = ["date", "total_amount"]
ALL_9_FIELDS  = ["date", "total_amount", "balance_due", "patient_name",
                 "philhealth_number", "diagnosis_code", "procedure_code",
                 "service_origin", "philhealth_benefit"]

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
    """Score a single eval file using standardized methodology."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    recs = [norm(p) for p in d.get("predictions", [])]
    
    # Bucket
    donut = [r for r in recs if "donut" in r["source_dataset"] or "invoices" in r["source_dataset"]]
    synth = [r for r in recs if r.get("stress_level") == "high" or r.get("tier") == "Specialized"]
    sroie = [r for r in recs if "sroie" in r["source_dataset"]]
    
    # --- Tier A: Donut only, FULL 9 fields ---
    ma = compute_metrics(donut) if donut else {}
    
    # --- Tier B: Synthetic noisy, Core Triad only ---
    mb = compute_metrics([filt(r, CORE_TRIAD) for r in synth]) if synth else {}
    
    # --- Tier B: Synthetic noisy, ALL 9 fields (raw, for comparison) ---
    mb_full = compute_metrics(synth) if synth else {}
    
    # --- SROIE: anchor fields only ---
    ms = compute_metrics([filt(r, ANCHOR_FIELDS) for r in sroie]) if sroie else {}
    
    # Per-field on Donut (Tier A)
    pf_a = ma.get("per_field", {})
    # Per-field on synthetic Core Triad (Tier B)
    pf_b = mb.get("per_field", {})
    # Per-field on synthetic full (for specialized/arithmetic)
    pf_b_full = mb_full.get("per_field", {})
    
    def g(m, k): return m.get(k, 0.0)
    def f1(pf, k): return pf.get(k, {}).get("f1", 0.0)
    
    return {
        # Tier A (Donut, full schema)
        "a_f1":       g(ma, "macro_f1"),
        "a_em":       g(ma, "fuzzy_exact_match"),
        "a_date":     f1(pf_a, "date"),
        "a_total":    f1(pf_a, "total_amount"),
        "a_patient":  f1(pf_a, "patient_name"),
        "a_balance":  f1(pf_a, "balance_due"),
        "a_phic":     f1(pf_a, "philhealth_number"),
        "a_diag":     f1(pf_a, "diagnosis_code"),
        "a_proc":     f1(pf_a, "procedure_code"),
        "a_origin":   f1(pf_a, "service_origin"),
        "a_benefit":  f1(pf_a, "philhealth_benefit"),
        # Tier B (Synthetic, Core Triad)
        "b_f1":       g(mb, "macro_f1"),
        "b_em":       g(mb, "fuzzy_exact_match"),
        "b_date":     f1(pf_b, "date"),
        "b_total":    f1(pf_b, "total_amount"),
        "b_patient":  f1(pf_b, "patient_name"),
        # Tier B full 9 (for specialized/arithmetic reporting)
        "b_balance":  f1(pf_b_full, "balance_due"),
        "b_phic":     f1(pf_b_full, "philhealth_number"),
        "b_diag":     f1(pf_b_full, "diagnosis_code"),
        "b_proc":     f1(pf_b_full, "procedure_code"),
        "b_origin":   f1(pf_b_full, "service_origin"),
        "b_benefit":  f1(pf_b_full, "philhealth_benefit"),
        # SROIE anchor
        "s_f1":       g(ms, "macro_f1"),
        "s_em":       g(ms, "fuzzy_exact_match"),
        # Counts
        "n_donut":    len(donut),
        "n_synth":    len(synth),
        "n_sroie":    len(sroie),
    }

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = {
    "Zero-Shot":       os.path.join(BASE, "eval_condition_FineTune_zero_shot.json"),
    "Logic E1.5":      os.path.join(BASE, "clean_result_raw", "eval_condition_clean_data_clean_data-epoch-1.5.json"),
    "Composed 97:3":   os.path.join(BASE, "clean_result_dual", "eval_condition_L9_N0.json"),
}

# Use the clean_result_dual version (1024px, older but correct eval) for Composed
# NOT the root version (1536px, slightly worse)

results = {}
for label, path in RUNS.items():
    m = score_file(path)
    results[label] = m
    if m:
        print(f"  {label}: OK  n={m['n_donut']}+{m['n_synth']}+{m['n_sroie']}={m['n_donut']+m['n_synth']+m['n_sroie']}  ({os.path.basename(path)})")
    else:
        print(f"  {label}: NOT FOUND")

cols = list(RUNS.keys())

def v(col, key):
    m = results.get(col)
    val = m.get(key) if m else None
    if val is None: return "  N/A"
    return f"{val:.4f}"

SEP = "=" * 88
SEP2 = "-" * 88

print(f"\n{SEP}")
print(f"  FeeVer 2.0 -- Validated Stratified IEEE Benchmark")
print(f"  All columns scored from raw predictions using identical compute_metrics")
print(SEP)
print(f"  {'Metric':<40} {'Zero-Shot':>10} {'Logic E1.5':>11} {'Composed':>10}  Gate")
print(SEP2)

rows = [
    ("TIER A: Clean Invoices (Donut)", None),
    ("  Macro F1",                       "a_f1"),
    ("  Fuzzy EM",                       "a_em"),
    ("  date",                           "a_date"),
    ("  total_amount",                   "a_total"),
    ("  patient_name",                   "a_patient"),
    ("  balance_due",                    "a_balance"),
    ("  philhealth_number",              "a_phic"),
    ("  diagnosis_code",                 "a_diag"),
    ("  procedure_code",                 "a_proc"),
    ("  service_origin",                 "a_origin"),
    ("  philhealth_benefit",             "a_benefit"),
    ("", None),
    ("TIER B: High-Noise Synthetic (Core Triad)", None),
    ("  Macro F1  (Core Triad)",         "b_f1"),
    ("  Fuzzy EM",                       "b_em"),
    ("  date",                           "b_date"),
    ("  total_amount",                   "b_total"),
    ("  patient_name",                   "b_patient"),
    ("", None),
    ("TIER B: High-Noise (All Fields, for reference)", None),
    ("  balance_due",                    "b_balance"),
    ("  philhealth_number",              "b_phic"),
    ("  diagnosis_code",                 "b_diag"),
    ("  procedure_code",                 "b_proc"),
    ("  service_origin",                 "b_origin"),
    ("  philhealth_benefit",             "b_benefit"),
    ("", None),
    ("ANCHOR: SROIE Receipts (date + total)", None),
    ("  Macro F1",                       "s_f1"),
    ("  Fuzzy EM",                       "s_em"),
]

for label, key in rows:
    if key is None:
        print(f"  {label}")
        continue
    vals = [v(col, key) for col in cols]
    # Gate markers
    gate_str = ""
    if key == "a_f1":
        try:
            g = "PASS" if float(vals[-1]) >= 0.94 else "FAIL"
            gate_str = f"  >= 0.94 [{g}]"
        except: pass
    elif key == "b_f1":
        try:
            g = "PASS" if float(vals[-1]) >= 0.75 else "FAIL"
            gate_str = f"  >= 0.75 [{g}]"
        except: pass
    elif key == "b_em":
        try:
            g = "PASS" if float(vals[-1]) > 0.30 else "FAIL"
            gate_str = f"  > 0.30 [{g}]"
        except: pass
    print(f"    {label:<38} {vals[0]:>10} {vals[1]:>11} {vals[2]:>10}{gate_str}")

print(SEP2)
print(f"  Source files:")
for label, path in RUNS.items():
    print(f"    {label}: {os.path.basename(path)}")
