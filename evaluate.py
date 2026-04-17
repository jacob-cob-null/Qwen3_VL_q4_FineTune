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

def normalize(v):
    if v is None: return None
    return str(v).lower().strip().replace("₱","").replace(",","")

def compute_metrics(results):
    fields = [
        "date", "patient_name", "philhealth_number",
        "diagnosis_code", "procedure_code",
        "total_amount", "philhealth_benefit",
        "balance_due"
    ]
    scores = {f: {"tp":0,"fp":0,"fn":0} for f in fields}
    exact  = 0

    for r in results:
        pred = r["prediction"]
        gt   = r["ground_truth"]
        if pred == gt: exact += 1
        for field in fields:
            pv = normalize(pred.get(field))
            gv = normalize(gt.get(field))
            if pv and gv:
                if pv == gv: scores[field]["tp"] += 1
                else:
                    scores[field]["fp"] += 1
                    scores[field]["fn"] += 1
            elif pv: scores[field]["fp"] += 1
            elif gv: scores[field]["fn"] += 1

    per_field = {}
    for field, c in scores.items():
        p = c["tp"]/(c["tp"]+c["fp"]) if c["tp"]+c["fp"] else 0
        r = c["tp"]/(c["tp"]+c["fn"]) if c["tp"]+c["fn"] else 0
        f1 = 2*p*r/(p+r) if p+r else 0
        per_field[field] = {"precision":round(p,4),"recall":round(r,4),"f1":round(f1,4)}

    macro_f1 = sum(v["f1"] for v in per_field.values()) / len(fields)
    return {
        "exact_match": round(exact/len(results),4) if len(results) else 0.0,
        "macro_f1":    round(macro_f1,4),
        "per_field":   per_field,
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

def load_sroie_test(max_samples=None):
    # Primary test set location (the actual directory the user confirmed)
    test_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Datasets", "Testing_Data", "sroie_2019_v2", "canonical"
    )
    test_file_path = os.path.join(test_dir, "test.jsonl")
    if not os.path.exists(test_file_path):
        # fallback: legacy path
        test_file_path = "Datasets/Testing_Data/sroie_2019_v2/canonical/test.jsonl"
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

def run_inference(model, tokenizer, sample):
    """
    Run Qwen3-VL inference on a single test sample.
    sample is a dict with keys: image_path, messages (chat-format), ground_truth.
    Returns the raw decoded string from the model.
    """
    import torch
    from PIL import Image as PILImage

    # Build the user message from the sample's messages list (user turn only)
    messages = sample.get("messages", [])
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return "{}"

    # Use the last user turn as the prompt
    user_content = user_messages[-1].get("content", [])

    # Load image if referenced — resolve the path to handle stale JSONL paths
    image_path = resolve_image_path(sample.get("image_path", ""))
    images = []
    if image_path and os.path.exists(image_path):
        try:
            images = [PILImage.open(image_path).convert("RGB")]
        except Exception as e:
            print(f"Warning: Failed to load eval image {image_path}: {e}")

    # Build the prompt for the model
    # Use only user turn — we want the model to generate the assistant response
    prompt_messages = [{"role": "user", "content": user_content}]

    try:
        # apply_chat_template with add_generation_prompt=True appends the assistant start token
        text_prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        return decoded.strip()

    except Exception as e:
        print(f"Inference error on sample: {e}")
        return "{}"


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

    sroie_test = load_sroie_test(max_samples=max_eval_samples)
    if not sroie_test:
        print("  No test samples found. Returning zero metrics.")
        return {"exact_match": 0.0, "macro_f1": 0.0, "per_field": {}}

    results = []
    for i, sample in enumerate(sroie_test):
        print(f"  Evaluating sample {i+1}/{len(sroie_test)}...", end="\r")
        raw_pred = run_inference(model, tokenizer, sample)
        pred_dict = safe_parse_json(raw_pred)
        gt_dict   = sample.get("ground_truth", {})  # Fixed: was sample.get("labels", {})
        results.append({
            "prediction":  pred_dict,
            "ground_truth": gt_dict,
            "raw_prediction": raw_pred,
        })

    print()  # newline after progress

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

if __name__ == "__main__":
    print("Testing compute_metrics...")
    mock_results = [{"prediction": {"date": "04/10/2026", "patient_name": "Juan"}, "ground_truth": {"date": "04/10/2026", "patient_name": "Juan Dela Cruz"}}]
    print(compute_metrics(mock_results))
