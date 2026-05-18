#!/usr/bin/env python3
"""
scripts/postprocess.py — Deterministic post-processing for FeeVer model predictions.

Applied to raw model output BEFORE scoring so that rule-based corrections
do not inflate the model's apparent learning — they simply repair known
structural gaps in the data sources (e.g. SROIE has no PhilHealth deduction).

Usage (standalone):
    python scripts/postprocess.py \
        --input eval_condition_FineTune_clean_data-epoch-2.5.json \
        --output eval_postprocessed.json

Integrated into evaluate.py automatically when imported:
    from scripts.postprocess import postprocess
"""

import json
import re
import argparse
import os
import sys

# All 8 target fields that every prediction must expose
TARGET_FIELDS = [
    "date",
    "patient_name",
    "philhealth_number",
    "diagnosis_code",
    "procedure_code",
    "service_origin",
    "total_amount",
    "philhealth_benefit",
    "balance_due",
]

# Sources where gross total == balance due (no PhilHealth deduction on the document)
_NON_MEDICAL_SOURCES = {"sroie_2019_v2", "cord_v2"}

# ISO date pattern YYYY-MM-DD
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _has_value(v) -> bool:
    """True when v is a meaningful non-null value."""
    if v is None:
        return False
    return str(v).strip().lower() not in ("", "null", "none", "n/a")


def _extract_year(date_str: str):
    """Return the 4-digit year string from a date field, or None."""
    if not date_str:
        return None
    m = _ISO_DATE_RE.search(str(date_str))
    if m:
        return m.group(1)
    # Try bare 4-digit year anywhere in the string
    m2 = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    return m2.group(0) if m2 else None


def postprocess(pred: dict, gt: dict, source: str) -> dict:
    """
    Apply deterministic corrections to raw model output before scoring.

    Corrections applied:
      1. Schema enforcement — ensure all TARGET_FIELDS keys exist (null if absent).
      2. balance_due fallback — for non-medical invoice sources (SROIE, Donut,
         CORD v2) the gross total IS the balance due; impute when balance_due
         is null and total_amount is present.
      3. Date anchor error flag — if model outputs a 2026 date but GT is a
         different year, set _date_anchor_error=True (diagnostic only; the
         erroneous prediction is still kept so the metric is honest).

    Args:
        pred:   Raw dict from model (may be missing keys).
        gt:     Ground-truth dict for this sample.
        source: Source dataset name (e.g. "paige_synthetic", "sroie_2019_v2").

    Returns:
        Corrected prediction dict (new object, original untouched).
    """
    out = dict(pred)

    # 1. Schema enforcement
    for field in TARGET_FIELDS:
        out.setdefault(field, None)

    # 2. balance_due fallback for non-medical sources
    if source in _NON_MEDICAL_SOURCES:
        if not _has_value(out.get("balance_due")) and _has_value(out.get("total_amount")):
            out["balance_due"] = out["total_amount"]
            out["_balance_due_imputed"] = True  # diagnostic flag

    # 3. 2026 Date anchor error flag (does NOT modify the prediction)
    pred_date = out.get("date")
    gt_date = gt.get("date") if gt else None
    if pred_date and gt_date:
        pred_year = _extract_year(str(pred_date))
        gt_year = _extract_year(str(gt_date))
        if pred_year == "2026" and gt_year and gt_year != "2026":
            out["_date_anchor_error"] = True

    return out


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _apply_to_eval_json(input_path: str, output_path: str) -> dict:
    """
    Load an eval_condition_*.json file, apply postprocess() to every prediction,
    and write the corrected file. Returns a summary dict.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    corrected = 0
    balance_due_imputed = 0
    date_anchor_errors = 0

    new_predictions = []
    for entry in predictions:
        source = entry.get("source", "unknown")
        pred = entry.get("pred", {})
        gt = entry.get("gt", {})

        new_pred = postprocess(pred, gt, source)

        if new_pred.get("_balance_due_imputed"):
            balance_due_imputed += 1
        if new_pred.get("_date_anchor_error"):
            date_anchor_errors += 1
        if new_pred != pred:
            corrected += 1

        new_entry = dict(entry)
        new_entry["pred"] = new_pred
        new_predictions.append(new_entry)

    data["predictions"] = new_predictions
    data["postprocess_summary"] = {
        "total_samples": len(predictions),
        "samples_corrected": corrected,
        "balance_due_imputed": balance_due_imputed,
        "date_anchor_errors": date_anchor_errors,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return data["postprocess_summary"]


def main():
    parser = argparse.ArgumentParser(
        description="Apply deterministic post-processing to FeeVer eval JSON output."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to eval_condition_*.json produced by evaluate.py",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the corrected JSON file",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    summary = _apply_to_eval_json(args.input, args.output)

    print(f"Post-processing complete: {args.input} -> {args.output}")
    print(json.dumps(summary, indent=2))
    print()
    if summary["balance_due_imputed"] > 0:
        print(
            f"  balance_due imputed for {summary['balance_due_imputed']} non-medical samples.\n"
            f"  Expected: balance_due F1 rises from ~0 to match total_amount F1 on SROIE/Donut."
        )
    if summary["date_anchor_errors"] > 0:
        print(
            f"  Date anchor errors (2026 overwrite) detected in {summary['date_anchor_errors']} samples.\n"
            f"  These are logged but NOT auto-corrected — metric stays honest."
        )


if __name__ == "__main__":
    main()
