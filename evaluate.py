import json
import os
import re
import sys

# Import from train.py for path resolution (graceful fallback if unavailable)
try:
    _train_dir = os.path.dirname(os.path.abspath(__file__))
    if _train_dir not in sys.path:
        sys.path.insert(0, _train_dir)
    from train import resolve_image_path
except Exception:
    def resolve_image_path(p):
        return p or ""

# Ensure multimodal messages contain image placeholders when images are present
try:
    from scripts.patch_prepare_multimodal import patch as _patch_prepare_multimodal
    _patch_prepare_multimodal()
except Exception:
    pass

# Unified extraction prompt — must match train.py exactly to prevent instruction drift
try:
    from scripts.prompts import EXTRACTION_PROMPT as _EXTRACTION_PROMPT
except Exception:
    _EXTRACTION_PROMPT = (
        "You are a document extraction assistant. "
        "Given the image of an invoice or billing document, "
        "extract the following fields and return them as a single JSON object "
        "with exactly these keys: "
        "date, patient_name, philhealth_number, diagnosis_code, procedure_code, "
        "total_amount, philhealth_benefit, balance_due. "
        "Use null for any field not present in the document. "
        "Return only the JSON object — no explanation, no markdown."
    )

import re as _re


# ---------------------------------------------------------------------------
# Field-aware normalization helpers
# ---------------------------------------------------------------------------

_AMOUNT_FIELDS = {"total_amount", "philhealth_benefit", "balance_due"}
_DATE_FIELDS   = {"date"}

# Text fields where CER adds partial-credit signal (amounts/dates use fuzzy numeric match instead)
_TEXT_FIELDS = set()

# Tier groupings for the IEEE success-matrix presentation
_FIELD_TIERS = {
    "temporal":  ["date"],
    "financial": ["total_amount", "balance_due"],
    "clinical":  [],
}


def _levenshtein(a, b):
    """Pure-Python Levenshtein edit distance — no external dependencies."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def _cer(pred_val, gt_val):
    """
    Character Error Rate for one field pair.
    CER = levenshtein(pred, gt) / len(gt), clamped to [0, 1].
    Both strings are normalised (lowercase, stripped) before comparison.
    Returns None when gt is null/empty — no signal to measure against.
    """
    if not _has_value(gt_val):
        return None
    gt_str   = normalize(str(gt_val)) or ""
    pred_str = normalize(str(pred_val)) if _has_value(pred_val) else ""
    if not gt_str:
        return None
    return round(min(1.0, _levenshtein(pred_str, gt_str) / len(gt_str)), 4)

# Date formats in rough frequency order for Philippine medical/invoice docs.
# Covers SROIE ("29 JUN 18", "11-APR-2018", "03 JUN 18"), Donut (MM/DD/YYYY),
# and ISO output from the model (YYYY-MM-DD already canonical).
_DATE_FORMATS = [
    # ISO (model output)
    "%Y-%m-%d",
    # DD/MM/YYYY and MM/DD/YYYY variants
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d/%m/%y", "%m/%d/%y",
    # Dash-separated
    "%d-%m-%Y", "%m-%d-%Y",
    "%d-%m-%y", "%m-%d-%y",
    # Dot-separated
    "%d.%m.%Y", "%m.%d.%Y",
    # Space-separated with abbreviated month name ("29 JUN 18", "03 JUN 18")
    "%d %b %Y", "%d %b %y",
    "%d %B %Y",
    # Abbreviated month with dash ("11-APR-2018", "27-MAR-2018")
    "%d-%b-%Y", "%d-%b-%y",
    # Month-first with name
    "%b %d %Y", "%b %d, %Y",
    "%B %d, %Y",
]

def _normalize_date(s):
    """Parse a date string to canonical YYYY-MM-DD. Falls back to stripped lowercase."""
    from datetime import datetime
    s = s.strip()
    # Try each format; .upper() so 'jun' and 'JUN' both work with %b
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.upper(), fmt.upper()).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s.lower()

def _strip_currency(s):
    # Also strip 'RM' prefix used in Malaysian receipts (SROIE dataset)
    s = _re.sub(r"^RM\s*", "", s.strip(), flags=_re.IGNORECASE)
    return _re.sub(r"[₱$€£¥\s,]", "", s).strip()

def _try_parse_float(s):
    try:
        return float(_strip_currency(s))
    except (ValueError, TypeError):
        return None

def normalize(v):
    """Basic normalization used for legacy exact-match checks."""
    if v is None:
        return None
    s = _re.sub(r"[₱$€£¥,]", "", str(v)).lower().strip()
    return s if s else None

def _has_value(v):
    """True when v is a meaningful non-null value (not None / empty / literal 'null')."""
    if v is None:
        return False
    return str(v).strip().lower() not in ("", "null", "none", "n/a")

def fields_match(field, pred_val, gt_val):
    """
    Return True if pred_val and gt_val are considered equivalent for this field.

    Dates  → both parsed to YYYY-MM-DD, then compared.
    Amounts→ numeric parse; match if within 2 % relative OR ±0.05 absolute
             (handles hospital rounding and minor OCR digit errors).
    Text   → case-insensitive, currency/comma-stripped string equality.
    """
    if not _has_value(pred_val) or not _has_value(gt_val):
        return False
    pv, gv = str(pred_val).strip(), str(gt_val).strip()

    if field in _DATE_FIELDS:
        return _normalize_date(pv) == _normalize_date(gv)

    if field in _AMOUNT_FIELDS:
        pf, gf = _try_parse_float(pv), _try_parse_float(gv)
        if pf is not None and gf is not None:
            if gf == 0:
                return pf == 0
            return (abs(pf - gf) / max(abs(gf), 1e-9)) <= 0.02 or abs(pf - gf) <= 0.05
        # Fall back to stripped string
        return normalize(pv) == normalize(gv)

    # Text fields (patient_name, codes, philhealth_number …)
    return normalize(pv) == normalize(gv)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results):
    """
    Compute precision / recall / F1 per field plus roll-up metrics.

    Key design choices:
    - null-vs-null pairs SKIPPED — no signal, avoids TP inflation.
    - macro_f1 averaged only over *active* fields (≥1 non-null GT value).
    - fuzzy_exact_match: every non-null GT field matches within field-aware tolerance.
    - strict exact_match retained for backward compatibility.
    - CER (Character Error Rate) added per text field for partial-credit signal.
    - tier_summary groups fields into temporal / financial / clinical buckets.
    """
    fields = [
        "date", "total_amount", "balance_due",
    ]
    scores = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields}
    # CER accumulators: sum of CER values and count of non-null GT pairs
    cer_sum   = {f: 0.0 for f in _TEXT_FIELDS}
    cer_count = {f: 0   for f in _TEXT_FIELDS}
    exact       = 0
    fuzzy_exact = 0

    for r in results:
        pred = r["prediction"]
        gt   = r["ground_truth"]

        # ── strict exact match ──────────────────────────────────────────
        if pred == gt:
            exact += 1

        # ── fuzzy exact: every non-null GT field within tolerance ────────
        gt_active = [f for f in fields if _has_value(gt.get(f))]
        if gt_active and all(fields_match(f, pred.get(f), gt.get(f)) for f in gt_active):
            fuzzy_exact += 1

        # ── per-field scoring + CER ──────────────────────────────────────
        for field in fields:
            pv, gv = pred.get(field), gt.get(field)
            has_pred = _has_value(pv)
            has_gt   = _has_value(gv)

            if not has_pred and not has_gt:
                continue  # both null → no signal

            if has_pred and has_gt:
                if fields_match(field, pv, gv):
                    scores[field]["tp"] += 1
                else:
                    scores[field]["fp"] += 1
                    scores[field]["fn"] += 1
            elif has_pred:
                scores[field]["fp"] += 1   # hallucinated
            else:
                scores[field]["fn"] += 1   # missed

            # CER for text fields (amounts/dates use fuzzy numeric match)
            if field in _TEXT_FIELDS:
                cer_val = _cer(pv, gv)
                if cer_val is not None:
                    cer_sum[field]   += cer_val
                    cer_count[field] += 1

    n = len(results)
    per_field = {}
    for field, c in scores.items():
        p  = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rc = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * p * rc / (p + rc)         if p + rc          else 0.0
        entry = {
            "precision": round(p,  4),
            "recall":    round(rc, 4),
            "f1":        round(f1, 4),
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }
        # Attach mean CER for text fields
        if field in _TEXT_FIELDS:
            cnt = cer_count[field]
            entry["mean_cer"] = round(cer_sum[field] / cnt, 4) if cnt else None
        per_field[field] = entry

    # Macro F1 — active fields only
    active = [f for f in fields if scores[f]["tp"] + scores[f]["fn"] > 0]
    macro_f1 = (
        sum(per_field[f]["f1"] for f in active) / len(active)
        if active else 0.0
    )

    # ── Tier summary (IEEE success-matrix format) ────────────────────────
    tier_summary = {}
    for tier, tier_fields in _FIELD_TIERS.items():
        tier_active = [f for f in tier_fields if f in active]
        if tier_active:
            tier_f1  = sum(per_field[f]["f1"]  for f in tier_active) / len(tier_active)
            tier_prec = sum(per_field[f]["precision"] for f in tier_active) / len(tier_active)
            tier_rec  = sum(per_field[f]["recall"]    for f in tier_active) / len(tier_active)
            # Mean CER for text-field tiers
            tier_text = [f for f in tier_active if f in _TEXT_FIELDS]
            mean_cer_vals = [
                per_field[f]["mean_cer"] for f in tier_text
                if per_field[f].get("mean_cer") is not None
            ]
            tier_summary[tier] = {
                "active_fields":  tier_active,
                "macro_f1":       round(tier_f1,  4),
                "macro_precision": round(tier_prec, 4),
                "macro_recall":   round(tier_rec,  4),
                "mean_cer":       round(sum(mean_cer_vals) / len(mean_cer_vals), 4)
                                  if mean_cer_vals else None,
            }
        else:
            tier_summary[tier] = {
                "active_fields": [],
                "macro_f1": None,
                "macro_precision": None,
                "macro_recall": None,
                "mean_cer": None,
                "note": "No GT coverage in this test set for this tier",
            }

    return {
        "exact_match":       round(exact       / n, 4) if n else 0.0,
        "fuzzy_exact_match": round(fuzzy_exact / n, 4) if n else 0.0,
        "macro_f1":          round(macro_f1,    4),
        "active_fields":     active,
        "per_field":         per_field,
        "tier_summary":      tier_summary,
    }


def safe_parse_json(text):
    """
    Extract and parse the first JSON object from a model output string.

    Handles three cases:
      1. Clean JSON  → standard parse.
      2. JSON embedded in surrounding text → regex-extracted then parsed.
      3. Truncated JSON (generation runaway/cutoff) → key-value salvage via
         regex so we don't lose all correctly-predicted fields in a sample.
    """
    if not text:
        return {}

    # ── 1. Direct parse ─────────────────────────────────────────────────
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # ── 2. Embedded JSON object ──────────────────────────────────────────
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # ── 3. Truncated JSON — salvage completed key-value pairs ────────────
    # Pattern: "key": "value" or "key": null (already-closed string values only)
    salvaged = {}
    # Match completed string values: "key": "value"
    for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*?)"', text):
        salvaged[m.group(1)] = m.group(2)
    # Match null values: "key": null
    for m in re.finditer(r'"(\w+)"\s*:\s*null', text):
        if m.group(1) not in salvaged:  # don't overwrite a real value
            salvaged[m.group(1)] = None
    # If we recovered at least one key, return the salvaged dict
    if salvaged:
        return salvaged

    return {}

def load_test_set(max_samples=None):
    """
    Load the evaluation test set.

    Priority order:
      1. mixed_test_226.jsonl  — canonical 3-source balanced set
                                  (100 CORD v2 + 26 Donut + 100 SROIE, seed 42)
      2. sroie_2019_v2 standalone canonical  — all images confirmed on disk
      3. all_sources_test.jsonl (merged)     — legacy fallback
    """
    _base = os.path.dirname(os.path.abspath(__file__))
    # Prefer an explicit eval manifest in workspace root (eval_tiered_v1.jsonl)
    candidates = [
        os.path.join(_base, "eval_tiered_v1.jsonl"),
        os.path.join(_base, "Datasets", "Testing_Data", "mixed_test_226.jsonl"),
        os.path.join(_base, "Datasets", "Testing_Data", "sroie_2019_v2", "canonical", "test.jsonl"),
        os.path.join(_base, "Datasets", "Training_Data", "golden", "merged", "all_sources_test.jsonl"),
    ]
    test_file_path = next((p for p in candidates if os.path.exists(p)), None)
    if test_file_path is None:
        print("Warning: no test set found. Returning empty eval list.")
        return []
    print(f"  Test set: {os.path.relpath(test_file_path, _base)}")
    samples = []
    with open(test_file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
            if max_samples and len(samples) >= max_samples:
                break
    return samples

# Keep old name as alias for backwards compatibility
load_sroie_test = load_test_set

def _prepare_sample_prompt(tokenizer, sample, max_image_size=None):
    """
    Build the text prompt string and load the image for one sample.
    Returns (text_prompt, image_or_None).

    max_image_size: if set, resize the image so its longest side is at most
    this many pixels before sending to the tokenizer. Fewer pixels = fewer
    visual patches = less VRAM. Good values: 448 (low), 672 (medium), 896 (high).
    """
    from PIL import Image as PILImage

    messages = sample.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return None, None

    # Override the user-turn text with the canonical prompt to prevent
    # instruction drift (mismatches between training and evaluation prompts).
    user_content = user_messages[-1].get("content", [])
    if isinstance(user_content, list):
        # Rebuild content: keep image tokens, replace all text items with unified prompt
        new_content = [c for c in user_content if isinstance(c, dict) and c.get("type") == "image"]
        # Ensure the canonical extraction prompt is present as the text item
        new_content.append({"type": "text", "text": _EXTRACTION_PROMPT})
        user_content = new_content

    image = None
    image_path = resolve_image_path(sample.get("image_path", ""))
    if image_path and os.path.exists(image_path):
        try:
            image = PILImage.open(image_path).convert("RGB")

            # Match training preprocessing: letterbox onto uniform canvas
            target_size = (768, 1024)
            pad_color = (245, 245, 245)

            # Resize to fit within target bounds while preserving aspect ratio
            try:
                image.thumbnail(target_size, PILImage.Resampling.LANCZOS)
            except Exception:
                image.thumbnail(target_size, PILImage.LANCZOS)

            # Create uniform canvas and paste centered
            canvas = PILImage.new('RGB', target_size, pad_color)
            offset = ((target_size[0] - image.width) // 2,
                      (target_size[1] - image.height) // 2)
            canvas.paste(image, offset)
            image = canvas

            # If the user content lacked an explicit image token, insert one
            if isinstance(user_content, list):
                has_image_item = any(isinstance(c, dict) and c.get("type") == "image" for c in user_content)
                if not has_image_item:
                    user_content = [{"type": "image"}] + user_content

        except Exception as e:
            print(f"Warning: Failed to load eval image {image_path}: {e}")

    # Build the prompt messages after we may have modified user_content above
    prompt_messages = [{"role": "user", "content": user_content}]

    text_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return text_prompt, image


def run_inference_batch(model, tokenizer, samples, batch_size=8, max_image_size=None):
    """
    Run batched Qwen3-VL inference over a list of samples.
    Returns a list of raw decoded strings (one per sample), in the same order.

    max_image_size: cap the longest image dimension before tokenization to
    reduce visual patch count and VRAM usage.
    """
    import torch

    device = next(model.parameters()).device
    all_decoded = []
    num_batches = (len(samples) + batch_size - 1) // batch_size

    # Use tqdm for a nice progress bar if available
    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(samples), desc="  Processing batches", unit="sample")
    except ImportError:
        pbar = None

    for batch_idx, batch_start in enumerate(range(0, len(samples), batch_size)):
        if pbar is None:
            print(f"  Batch {batch_idx + 1}/{num_batches} (samples {batch_start + 1}-{min(batch_start + batch_size, len(samples))})...", end="\r")
        
        batch = samples[batch_start : batch_start + batch_size]

        # --- Prepare prompts and images for the whole batch ---
        texts, images_per_sample = [], []
        valid_mask = []  # track which samples produced a valid prompt
        for sample in batch:
            text, img = _prepare_sample_prompt(tokenizer, sample, max_image_size=max_image_size)
            if text is None:
                # No user turn — emit empty result
                valid_mask.append(False)
                texts.append("")          # placeholder, won't be sent to GPU
                images_per_sample.append(None)
            else:
                valid_mask.append(True)
                texts.append(text)
                images_per_sample.append(img)

        # Filter to only valid items for this batch
        valid_texts  = [t for t, v in zip(texts,  valid_mask) if v]
        valid_images = [i for i, v in zip(images_per_sample, valid_mask) if v]

        if not valid_texts:
            all_decoded.extend(["{}" for _ in batch])
            continue

        # Qwen3-VL processor expects a flat list of PIL images (one per sample
        # that has an image). Samples without images are passed as None.
        try:
            inputs = tokenizer(
                text=valid_texts,
                images=valid_images if any(img is not None for img in valid_images) else None,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=150,   # 8-field JSON is well under this
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens for each item in the batch
            input_len = inputs["input_ids"].shape[1]
            batch_decoded = [
                tokenizer.decode(output_ids[i, input_len:], skip_special_tokens=True).strip()
                for i in range(len(valid_texts))
            ]
        except Exception as e:
            print(f"\nBatch inference error (batch starting at {batch_start}): {e}")
            batch_decoded = ["{}" for _ in valid_texts]

        # Re-insert empty results for invalid samples, preserving original order
        decoded_iter = iter(batch_decoded)
        for v in valid_mask:
            all_decoded.append(next(decoded_iter) if v else "{}")
        
        if pbar:
            pbar.update(len(batch))

    if pbar:
        pbar.close()
    else:
        print() # Newline after the \r progress line

    return all_decoded


def evaluate_condition(condition_id, adapter_path, model=None, tokenizer=None, max_eval_samples=None):
    """
    Evaluate a trained model on the SROIE test set.

    The model passed in is already LoRA-adapted and trained — we do NOT
    re-load or re-wrap the adapter here to avoid the double-PEFT-wrapping
    warning. We simply switch it to inference mode and run generate().

    Args:
        condition_id:      Experiment condition label (e.g. "A").
        adapter_path:      Path where the adapter was saved (unused for loading
                           since the model is already in memory, kept for logging).
        model:             The trained PeftModel, still in memory from training.
        tokenizer:         The tokenizer used during training.
        max_eval_samples:  Cap the number of test samples (useful for smoke tests).
    """
    if model is None or tokenizer is None:
        print("Model or tokenizer not provided for evaluation. Skipping actual inference.")
        return {"exact_match": 0.0, "macro_f1": 0.0, "per_field": {}}

    # Switch to inference mode (Unsloth patch for 2x faster generation)
    try:
        from unsloth import FastVisionModel
        model = FastVisionModel.for_inference(model)
    except Exception as e:
        print(f"Note: FastVisionModel.for_inference unavailable ({e}), continuing without it.")

    print(f"  Adapter saved at: {adapter_path}")
    print(f"  Loading test set{'(capped at ' + str(max_eval_samples) + ' samples)' if max_eval_samples else ''}...")

    sroie_test = load_test_set(max_samples=max_eval_samples)
    # Allow CLI override for test manifest if provided
    if args.test_manifest:
        tm = os.path.abspath(args.test_manifest)
        if os.path.exists(tm):
            print(f"  Overriding test set with {tm}")
            with open(tm, 'r', encoding='utf-8') as f:
                sroie_test = [json.loads(l) for l in f if l.strip()]
                if max_eval_samples:
                    sroie_test = sroie_test[:max_eval_samples]
        else:
            print(f"  Warning: --test-manifest {tm} not found; using discovered test set.")
    if not sroie_test:
        print("  No test samples found. Returning zero metrics.")
        return {"exact_match": 0.0, "macro_f1": 0.0, "per_field": {}}

    batch_size = getattr(evaluate_condition, "_batch_size", 8)
    max_image_size = getattr(evaluate_condition, "_max_image_size", None)

    img_size_str = f", max_image_size={max_image_size}px" if max_image_size else ""
    print(f"  Running batched inference (batch_size={batch_size}{img_size_str}) over {len(sroie_test)} samples...")

    raw_preds = run_inference_batch(model, tokenizer, sroie_test, batch_size=batch_size, max_image_size=max_image_size)

    results = []
    for i, (raw_pred, sample) in enumerate(zip(raw_preds, sroie_test)):
        pred_dict = safe_parse_json(raw_pred)
        gt_dict   = sample.get("ground_truth", {})
        results.append({
            "prediction":    pred_dict,
            "ground_truth":  gt_dict,
            "raw_prediction": raw_pred,
            "source_dataset": sample.get("source_dataset", "unknown"),
        })

    print(f"  Inference complete ({len(results)} samples).")

    metrics = compute_metrics(results)

    # ── Per-source breakdown ─────────────────────────────────────────────
    sources = sorted({r["source_dataset"] for r in results})
    per_source = {}
    for src in sources:
        subset = [r for r in results if r["source_dataset"] == src]
        per_source[src] = {
            "num_samples": len(subset),
            "metrics":     compute_metrics(subset),
        }
        print(f"  [{src}] n={len(subset)}  "
              f"macro_f1={per_source[src]['metrics']['macro_f1']:.4f}  "
              f"fuzzy_em={per_source[src]['metrics']['fuzzy_exact_match']:.4f}")

    # ── Domain grouping (Tiered reporting) ───────────────────────────────
    DOMAIN_GROUPS = {
        "Tier A (Invoice-Style)": ["sroie_2019_v2", "invoices_donut_v1"],
        "Tier B (Edge Cases - CORD)": ["cord_v2"],
    }
    per_domain = {}
    for dname, d_sources in DOMAIN_GROUPS.items():
        subset = [r for r in results if r["source_dataset"] in d_sources]
        if subset:
            per_domain[dname] = {
                "num_samples": len(subset),
                "metrics": compute_metrics(subset),
                "sources": sorted(set(r["source_dataset"] for r in subset)),
            }
            m = per_domain[dname]["metrics"]
            print(f"  [{dname}] n={len(subset)}  macro_f1={m['macro_f1']:.4f}  fuzzy_em={m['fuzzy_exact_match']:.4f}")
        else:
            per_domain[dname] = {"num_samples": 0, "metrics": None, "sources": []}

    out_path = f"eval_condition_{condition_id}.json"
    with open(out_path, "w") as f:
        json.dump({
            "condition_id": condition_id,
            "adapter_path": adapter_path,
            "num_samples":  len(results),
            "metrics":      metrics,
            "per_source":   per_source,
            "per_domain":   per_domain,
            "predictions":  [
                {
                    "source": r["source_dataset"],
                    "gt":     r["ground_truth"],
                    "pred":   r["prediction"],
                    "raw":    r["raw_prediction"],
                }
                for r in results
            ],
        }, f, indent=2)
    print(f"  Eval results saved to {out_path}")

    return metrics

def _load_model_for_eval(adapter_path):
    """
    Load the base model + LoRA adapter from disk for standalone evaluation.
    This is only called when running evaluate.py directly (not from train.py).
    """
    import os
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    import torch
    # Provide fallback for float8 dtype names expected by PEFT/unsloth
    try:
        for _name in ["float8_e8m0fnu", "float8_e4m3fn", "float8_e5m2"]:
            if not hasattr(torch, _name):
                setattr(torch, _name, torch.bfloat16)
        if not hasattr(torch, "__orig_getattr__"):
            orig_getattr = getattr(torch, "__getattr__", None)
            def _torch_getattr(name):
                if "float8" in name:
                    return torch.bfloat16
                if orig_getattr:
                    return orig_getattr(name)
                raise AttributeError(f"module 'torch' has no attribute '{name}'")
            setattr(torch, "__orig_getattr__", orig_getattr)
            setattr(torch, "__getattr__", _torch_getattr)
    except Exception:
        pass

    from unsloth import FastVisionModel

    print(f"  Loading base model + adapter from: {adapter_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=adapter_path,  # points to saved adapter dir
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )

    # Apply fixes from train.py to ensure consistency
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if actual_tokenizer.pad_token is None:
        actual_tokenizer.pad_token = "<|endoftext|>"
        
    if len(actual_tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(actual_tokenizer))

    print("  Enabling Unsloth FastVisionModel inference optimization...")
    model = FastVisionModel.for_inference(model)
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    import gc

    # Map condition IDs to their adapter directories (mirrors train.py naming)
    CONDITION_DIRS = {
        "A": "paige-lora-condition-A",
        "B": "paige-lora-condition-B",
        "C": "paige-lora-condition-C",
        "D": "paige-lora-condition-D",
        "E": "paige-lora-condition-E",
        "F": "paige-lora-condition-F",
        "G": "paige-lora-condition-G",
        "clean_data": "paige-lora-condition-clean_data",
        "smoketest": "paige-smoketest",
    }

    parser = argparse.ArgumentParser(description="pAIge standalone evaluation")
    parser.add_argument(
        "--id", nargs="+", required=True,
        help="Condition letter(s) to evaluate, e.g. --id A B G (or 'smoketest' or 'ALL')",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap the number of test samples (useful for quick checks)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Inference batch size (default 4; lower if you hit OOM, raise if VRAM allows)",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=896,
        help="Resize images so the longest side is at most this many pixels before inference "
             "(default 896). Synced with training resolution. "
             "Use 448 or 672 to trade accuracy for VRAM.",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Override the adapter directory (single condition only)",
    )
    parser.add_argument(
        "--test-manifest", type=str, default=None,
        help="Path to a test manifest (.jsonl). If provided, it overrides automatic test set discovery.",
    )
    parser.add_argument(
        "--results-dir", type=str, default=".",
        help="Root directory containing paige-lora-condition-* folders. "
             "E.g. --results-dir results/1_epoch or --results-dir results/scale_epoch",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    # Derive a short label from the last path component for output file naming
    # e.g.  results/1_epoch  -> '1_epoch'   |   '.'  -> 'default'
    _rdir_label = os.path.basename(results_dir) or "default"

    # Propagate settings to evaluate_condition via function attributes
    evaluate_condition._batch_size = args.batch_size
    evaluate_condition._max_image_size = args.max_image_size

    # Keep raw ids but compare case-insensitively against known condition names
    targets = args.id
    # Build a normalized map of CONDITION_DIRS for case-insensitive lookup
    cond_map = {k.lower(): v for k, v in CONDITION_DIRS.items()}

    expanded_targets = []
    if any(t.lower() == "all" for t in targets):
        if os.path.exists(results_dir):
            for item in sorted(os.listdir(results_dir)):
                full_path = os.path.join(results_dir, item)
                if os.path.isdir(full_path):
                    # Verify it has model/adapter config
                    if os.path.exists(os.path.join(full_path, "adapter_config.json")) or os.path.exists(os.path.join(full_path, "config.json")):
                        cond_id = item
                        if cond_id.startswith("paige-lora-condition-"):
                            cond_id = cond_id[len("paige-lora-condition-"):]
                        expanded_targets.append((cond_id, full_path))
        if not expanded_targets:
            print(f"No adapters found in '{results_dir}'.")
    else:
        for cond_id in targets:
            if args.adapter_path:
                expanded_targets.append((cond_id, args.adapter_path))
                continue
            cond_key = cond_id.lower()
            if cond_key in cond_map:
                base_name = cond_map[cond_key]
                base_path = os.path.join(results_dir, base_name)
                found = False
                if os.path.isdir(base_path):
                    expanded_targets.append((cond_id, base_path))
                    found = True

                # Autodetect epoch steps if any
                if os.path.exists(results_dir):
                    for item in sorted(os.listdir(results_dir)):
                        if item.startswith(base_name + "-epoch-") and os.path.isdir(os.path.join(results_dir, item)):
                            suffix = item[len(base_name):]  # e.g., '-epoch-1.5'
                            expanded_targets.append((cond_id + suffix, os.path.join(results_dir, item)))
                            found = True

                if not found:
                    print(f"Adapter not found for condition '{cond_id}' in '{results_dir}'.")
            else:
                print(f"Unknown condition '{cond_id}'. Valid options: {list(CONDITION_DIRS.keys())} or 'ALL'")

    for cond_id, adapter_path in expanded_targets:
        if not os.path.isdir(adapter_path):
            print(f"Adapter not found at '{adapter_path}'. Skipping.")
            continue

        # Label used in output filename: e.g. '1_epoch_A'  or  'scale_epoch_C'
        output_label = f"{_rdir_label}_{cond_id}" if _rdir_label != "default" else cond_id

        print(f"\n{'='*60}\nEVALUATING [{_rdir_label}] Condition {cond_id}\n{'='*60}")
        model, tokenizer = _load_model_for_eval(adapter_path)

        metrics = evaluate_condition(
            condition_id=output_label,
            adapter_path=adapter_path,
            model=model,
            tokenizer=tokenizer,
            max_eval_samples=args.max_samples,
        )
        print(f"Metrics for [{_rdir_label}] condition {cond_id}: {metrics}")

        # Release VRAM before evaluating the next condition
        del model, tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"VRAM cleared after condition {cond_id}.\n")
