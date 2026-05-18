import json

STRUCTURAL_F1_GATE = 0.94
SPECIALIZED_EM_GATE = 0.30

d = json.load(open('eval_condition_L8_N2.json'))
per_source = d.get('per_source', {})

structural_sources = ['invoices_donut_v1']
vals = [per_source[s]['metrics']['macro_f1'] for s in structural_sources if s in per_source]
structural_f1 = sum(vals) / len(vals) if vals else 0.0
spec_em = per_source.get('paige_synthetic', {}).get('metrics', {}).get('fuzzy_exact_match', 0.0)
spec_f1 = per_source.get('paige_synthetic', {}).get('metrics', {}).get('macro_f1', 0.0)

passes_struct = structural_f1 >= STRUCTURAL_F1_GATE
passes_em = spec_em > SPECIALIZED_EM_GATE
passes = passes_struct and passes_em

print("=== 80:20 Composed (strong-filter eval) — Corrected Gate ===")
print(f"  struct_f1 (donut only) = {structural_f1:.4f}  gate={STRUCTURAL_F1_GATE}  {'PASS' if passes_struct else 'FAIL'}")
print(f"  spec_f1   (synthetic)  = {spec_f1:.4f}")
print(f"  spec_em   (synthetic)  = {spec_em:.4f}  gate={SPECIALIZED_EM_GATE}  {'PASS' if passes_em else 'FAIL'}")
print(f"  OVERALL: {'PASS' if passes else 'FAIL'}")
print()
print("Per-source breakdown:")
for src, info in per_source.items():
    m = info['metrics']
    print(f"  [{src}]  n={info['num_samples']}  macro_f1={m['macro_f1']:.4f}  fuzzy_em={m['fuzzy_exact_match']:.4f}")
