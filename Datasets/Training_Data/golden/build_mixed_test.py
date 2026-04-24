"""
Build the canonical mixed evaluation test set for pAIge IEEE ablation.

Composition (Option D — pure held-out only):
  - 100 CORD v2 test  (all 100 available; already held-out by HF split)
  -  26 Invoices Donut v1 test  (all 26 available; only clean held-out partition)
  - 100 SROIE 2019 v2 test  (random 100 of 347)
  -----------
  226 total, shuffled with fixed seed for reproducibility

Output:
  Datasets/Testing_Data/mixed_test_226.jsonl
  (each record has: sample_id, source_dataset, image_path [resolved], ground_truth, messages)
"""

import json
import os
import random
import sys

# Script lives at Datasets/Training_Data/golden/ — 3 levels below workspace root.
# Use CWD (we always invoke from FineTune/) for robustness.
BASE = os.getcwd()
SEED   = 42
TARGET = 226

# ── Source JSONL paths ──────────────────────────────────────────────────────
SOURCES = {
    "cord_v2": {
        "jsonl": os.path.join(BASE, "Datasets/Training_Data/golden/sources/cord_v2/canonical/test.jsonl"),
        "img_root": os.path.join(BASE, "Datasets/Training_Data/golden/sources/cord_v2/images/test"),
        "n": 100,
    },
    "invoices_donut_v1": {
        "jsonl": os.path.join(BASE, "Datasets/Training_Data/golden/sources/invoices_donut_v1/canonical/test.jsonl"),
        "img_root": os.path.join(BASE, "Datasets/Training_Data/golden/sources/invoices_donut_v1/images/test"),
        "n": None,  # take all
    },
    "sroie_2019_v2": {
        "jsonl": os.path.join(BASE, "Datasets/Testing_Data/sroie_2019_v2/canonical/test.jsonl"),
        "img_root": os.path.join(BASE, "Datasets/Testing_Data/sroie_2019_v2/images/test"),
        "n": 100,
    },
}

OUT_PATH = os.path.join(BASE, "Datasets/Testing_Data/mixed_test_226.jsonl")

# ── Image resolver ──────────────────────────────────────────────────────────
def resolve_image(record, img_root):
    """Return absolute path to image or None."""
    ip = record.get("image_path", "")
    fname = os.path.basename(ip) if ip else ""

    # 1. Absolute and exists
    if ip and os.path.isabs(ip) and os.path.exists(ip):
        return ip
    # 2. Relative to BASE
    if ip:
        cand = os.path.join(BASE, ip)
        if os.path.exists(cand):
            return cand
    # 3. Basename in img_root
    if fname:
        cand = os.path.join(img_root, fname)
        if os.path.exists(cand):
            return cand
        # Walk img_root for subdirs
        for dirpath, _, files in os.walk(img_root):
            if fname in files:
                return os.path.join(dirpath, fname)
    return None


# ── Load each source ─────────────────────────────────────────────────────────
rng = random.Random(SEED)
all_records = []
stats = {}

for src_name, cfg in SOURCES.items():
    jsonl_path = cfg["jsonl"]
    img_root   = cfg["img_root"]
    n_target   = cfg["n"]

    if not os.path.exists(jsonl_path):
        print(f"  ERROR: JSONL not found: {jsonl_path}")
        sys.exit(1)

    pool = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                pool.append(json.loads(line))
            except Exception:
                continue

    # Resolve images and filter to records with present images
    resolved_pool = []
    for r in pool:
        img = resolve_image(r, img_root)
        if img:
            r["image_path"] = img          # overwrite with absolute path
            r["source_dataset"] = src_name # ensure source tag is present
            resolved_pool.append(r)

    missing = len(pool) - len(resolved_pool)
    if missing:
        print(f"  Warning: {src_name} — {missing} records dropped (no image on disk)")

    # Sample
    if n_target is None or n_target >= len(resolved_pool):
        selected = resolved_pool
    else:
        selected = rng.sample(resolved_pool, n_target)

    stats[src_name] = len(selected)
    all_records.extend(selected)
    print(f"  {src_name}: {len(selected)} records selected (pool had {len(resolved_pool)} with images)")

# ── Shuffle the combined set ─────────────────────────────────────────────────
rng.shuffle(all_records)

# ── Write output ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as fh:
    for r in all_records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n{'='*55}")
print(f"Mixed test set written to: {OUT_PATH}")
print(f"Total records: {len(all_records)}")
for src, n in stats.items():
    print(f"  {src}: {n}")
print(f"Shuffle seed: {SEED}")

# ── Quick GT quality report ──────────────────────────────────────────────────
FIELDS = ["date", "patient_name", "philhealth_number", "diagnosis_code",
          "procedure_code", "total_amount", "philhealth_benefit", "balance_due"]
print(f"\n{'-'*55}")
print("GT field coverage (non-null) across full mixed set:")
for f in FIELDS:
    count = sum(1 for r in all_records if r.get("ground_truth", {}).get(f) is not None)
    pct = 100 * count / len(all_records) if all_records else 0
    bar = "#" * int(pct / 5)
    print(f"  {f:<22} {count:>3}/{len(all_records)}  {pct:5.1f}%  {bar}")
