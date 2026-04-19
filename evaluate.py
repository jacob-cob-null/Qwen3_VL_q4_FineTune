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

# Date formats in rough frequency order for Philippine medical/invoice docs
_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
    "%d.%m.%Y", "%m.%d.%Y",
    "%d/%m/%y", "%m/%d/%y",
    "%d-%m-%y", "%m-%d-%y",
    "%d %b %Y", "%d %B %Y",
    "%b %d %Y", "%B %d, %Y",
    "%d %b %y", "%b %d %y",
]

def _normalize_date(s):
    """Parse a date string to canonical YYYY-MM-DD. Falls back to stripped lowercase."""
    from datetime import datetime
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s.upper(), fmt.replace("%b", "%b").upper()).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s.lower()

def _strip_currency(s):
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

    Key design choices vs the naive implementation:
    - null-vs-null pairs are SKIPPED — they carry no signal and would
      inflate TP counts on datasets where most fields are absent.
    - macro_f1 is averaged only over *active* fields (fields that have at
      least one non-null GT value in the test set).  Always-null fields
      (e.g. philhealth_number on CORD receipts) no longer dilute the macro.
    - fuzzy_exact_match: a sample qualifies if every non-null GT field
      matches within the field-aware tolerance above.
    - strict exact_match is retained unchanged for backward compatibility.
    """
    fields = [
        "date", "patient_name", "philhealth_number",
        "diagnosis_code", "procedure_code",
        "total_amount", "philhealth_benefit",
        "balance_due",
    ]
    scores = {f: {"tp": 0, "fp": 0, "fn": 0} for f in fields}
    exact       = 0
    fuzzy_exact = 0

    for r in results:
        pred = r["prediction"]
        gt   = r["ground_truth"]

        # ── strict exact match (unchanged) ──────────────────────────────
        if pred == gt:
            exact += 1

        # ── fuzzy exact: every non-null GT field matches within tolerance ─
        gt_active = [f for f in fields if _has_value(gt.get(f))]
        if gt_active and all(fields_match(f, pred.get(f), gt.get(f)) for f in gt_active):
            fuzzy_exact += 1

        # ── per-field scoring ────────────────────────────────────────────
        for field in fields:
            pv, gv = pred.get(field), gt.get(field)
            has_pred = _has_value(pv)
            has_gt   = _has_value(gv)

            if not has_pred and not has_gt:
                continue  # both null → no signal, skip

            if has_pred and has_gt:
                if fields_match(field, pv, gv):
                    scores[field]["tp"] += 1
                else:
                    scores[field]["fp"] += 1
                    scores[field]["fn"] += 1
            elif has_pred:
                scores[field]["fp"] += 1   # hallucinated a value
            else:
                scores[field]["fn"] += 1   # missed a value

    n = len(results)
    per_field = {}
    for field, c in scores.items():
        p  = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        rc = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f1 = 2 * p * rc / (p + rc)         if p + rc          else 0.0
        per_field[field] = {
            "precision": round(p,  4),
            "recall":    round(rc, 4),
            "f1":        round(f1, 4),
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
        }

    # Average only over fields that had at least one non-null GT value
    active = [f for f in fields if scores[f]["tp"] + scores[f]["fn"] > 0]
    macro_f1 = (
        sum(per_field[f]["f1"] for f in active) / len(active)
        if active else 0.0
    )

    return {
        "exact_match":       round(exact       / n, 4) if n else 0.0,
        "fuzzy_exact_match": round(fuzzy_exact / n, 4) if n else 0.0,
        "macro_f1":          round(macro_f1,    4),
        "active_fields":     active,   # which fields drove the macro_f1
        "per_field":         per_field,
    }


def safe_parse_json(text):
    """Extract and parse the first JSON object from a model output string."""
    if not text:
        return {}
    # Direct parse first
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Try to extract a JSON object embedded in surrounding text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}

def load_test_set(max_samples=None):
    """Load the merged evaluation test set (cord_v2 + invoices_donut_v1)."""
    # Primary: merged test JSONL built by organize_golden_jsonl.py
    test_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Datasets", "Training_Data", "golden", "merged", "all_sources_test.jsonl"
    )
    if not os.path.exists(test_file_path):
        # Fallback: old SROIE location (graceful degradation)
        test_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Datasets", "Testing_Data", "sroie_2019_v2", "canonical", "test.jsonl"
        )
    if not os.path.exists(test_file_path):
        print(f"Warning: test set not found at {test_file_path}. Returning empty eval list.")
        return []
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
        new_content.append({"type": "text", "text": _EXTRACTION_PROMPT})
        user_content = new_content
    prompt_messages = [{"role": "user", "content": user_content}]

    text_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image = None
    image_path = resolve_image_path(sample.get("image_path", ""))
    if image_path and os.path.exists(image_path):
        try:
            image = PILImage.open(image_path).convert("RGB")
            if max_image_size is not None:
                w, h = image.size
                longest = max(w, h)
                if longest > max_image_size:
                    scale = max_image_size / longest
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = image.resize((new_w, new_h), PILImage.LANCZOS)
        except Exception as e:
            print(f"Warning: Failed to load eval image {image_path}: {e}")

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
            "prediction":   pred_dict,
            "ground_truth": gt_dict,
            "raw_prediction": raw_pred,
        })

    print(f"  Inference complete ({len(results)} samples).")

    metrics = compute_metrics(results)

    out_path = f"eval_condition_{condition_id}.json"
    with open(out_path, "w") as f:
        json.dump({
            "condition_id": condition_id,
            "adapter_path": adapter_path,
            "num_samples":  len(results),
            "metrics":      metrics,
            "predictions":  [
                {"gt": r["ground_truth"], "pred": r["prediction"], "raw": r["raw_prediction"]}
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
        "A": "./paige-lora-condition-A",
        "B": "./paige-lora-condition-B",
        "C": "./paige-lora-condition-C",
        "D": "./paige-lora-condition-D",
        "E": "./paige-lora-condition-E",
        "F": "./paige-lora-condition-F",
        "G": "./paige-lora-condition-G",
        "smoketest": "./paige-smoketest",
    }

    parser = argparse.ArgumentParser(description="pAIge standalone evaluation")
    parser.add_argument(
        "--id", nargs="+", required=True,
        help="Condition letter(s) to evaluate, e.g. --id A B G (or 'smoketest')",
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
    args = parser.parse_args()

    # Propagate settings to evaluate_condition via function attributes
    evaluate_condition._batch_size = args.batch_size
    evaluate_condition._max_image_size = args.max_image_size

    targets = [i.upper() if i.lower() != "smoketest" else "smoketest" for i in args.id]

    for cond_id in targets:
        if args.adapter_path:
            adapter_path = args.adapter_path
        elif cond_id in CONDITION_DIRS:
            adapter_path = CONDITION_DIRS[cond_id]
        else:
            print(f"Unknown condition '{cond_id}'. Valid options: {list(CONDITION_DIRS.keys())}")
            continue

        if not os.path.isdir(adapter_path):
            print(f"Adapter not found at '{adapter_path}'. Train condition {cond_id} first.")
            continue

        print(f"\n{'='*60}\nEVALUATING Condition {cond_id}\n{'='*60}")
        model, tokenizer = _load_model_for_eval(adapter_path)

        metrics = evaluate_condition(
            condition_id=cond_id,
            adapter_path=adapter_path,
            model=model,
            tokenizer=tokenizer,
            max_eval_samples=args.max_samples,
        )
        print(f"Metrics for condition {cond_id}: {metrics}")

        # Release VRAM before evaluating the next condition
        del model, tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"VRAM cleared after condition {cond_id}.\n")
