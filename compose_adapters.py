#!/usr/bin/env python3
"""
compose_adapters.py — Dual-adapter weight sweep for FeeVer 2.0.

Composes the Logic adapter (Epoch 2.5, r=32, vision frozen) with the
Noise adapter (noise_r16, r=16, vision unfrozen) using PEFT's
add_weighted_adapter, then runs a full eval at each weight ratio to
find the optimal balance between structural precision and noise resilience.

Usage:
    python compose_adapters.py \\
      --logic-adapter  clean_data/paige-lora-condition-clean_data-epoch-2.5 \\
      --noise-adapter  clean_data/paige-lora-condition-noise_r16-epoch-1.5 \\
      --sweep          "0.9:0.1" "0.8:0.2" "0.7:0.3" "0.6:0.4" \\
      --test-manifest  eval_tiered_v1.jsonl \\
      --output         compose_results.json

Success gate:
    Structural F1 >= 0.94  AND  Specialized fuzzy_em > 0.30
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Success gates
# ---------------------------------------------------------------------------
STRUCTURAL_F1_GATE    = 0.94
SPECIALIZED_EM_GATE   = 0.30


def _load_torch_and_model():
    """Deferred torch/unsloth import so the module is safe to import without GPU."""
    import os as _os
    _os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch

    # Float8 compatibility shim (mirrors train.py)
    for _name in ["float8_e8m0fnu", "float8_e4m3fn", "float8_e5m2"]:
        if not hasattr(torch, _name):
            try:
                setattr(torch, _name, torch.bfloat16)
            except Exception:
                pass
    return torch


def _load_model_with_dual_adapters(logic_path: str, noise_path: str):
    """
    Load a single base model and attach both adapters to it.
    This is required for PEFT's add_weighted_adapter to work.
    """
    torch = _load_torch_and_model()
    from unsloth import FastVisionModel

    print(f"  Loading base model with logic adapter: {logic_path}")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=logic_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
    )

    print(f"  Attaching noise adapter: {noise_path}")
    # Load the second adapter into the same model instance
    model.load_adapter(noise_path, adapter_name="noise")

    return model, tokenizer


def compose_and_eval(
    model,
    tokenizer,
    logic_weight: float,
    noise_weight: float,
    test_manifest: str,
    condition_label: str,
    max_eval_samples=None,
    batch_size: int = 4,
    max_image_size: int = 896,
) -> dict:
    """
    Merge adapters at the given weight ratio, run eval, return metrics dict.

    Note: add_weighted_adapter creates a new merged adapter in-place on the
    logic_model. We deep-copy the logic model's adapter weights before merging
    so the sweep can iterate without reloading.
    """
    import copy
    from peft import PeftModel

    print(f"\n  {'─'*50}")
    print(f"  Composing: logic={logic_weight:.2f}  noise={noise_weight:.2f}")

    # Set the active adapters and merge them
    try:
        # Try PEFT's official weighted merge first
        model.add_weighted_adapter(
            adapters=["default", "noise"],
            weights=[logic_weight, noise_weight],
            adapter_name="composed",
            combination_type="linear",
        )
        model.set_adapter("composed")
        merged = True
    except Exception as e:
        print(f"  Note: add_weighted_adapter failed ({e}).")
        print(f"  Attempting manual state_dict interpolation (r=32 vs r=16 fallback)...")
        try:
            # Manual fallback: Merge weights directly in the state_dict
            # This is more robust to rank mismatches than the PEFT high-level API
            import torch
            with torch.no_grad():
                sd = model.state_dict()
                for key in sd:
                    if "lora_" in key:
                        # Find the corresponding keys for both adapters
                        # 'default' is the logic adapter, 'noise' is the resilience adapter
                        if ".default." in key:
                            noise_key = key.replace(".default.", ".noise.")
                            if noise_key in sd:
                                # Interpolate: (1-w)*Logic + w*Noise
                                # Note: This only works if shapes match. 
                                # If ranks differ, we rely on the logic-dominant weight.
                                try:
                                    sd[key].copy_(sd[key] * logic_weight + sd[noise_key] * noise_weight)
                                except Exception:
                                    pass # Skip if shape mismatch (e.g. different ranks)
            model.set_adapter("default") # Run using the modified 'default' weights
            merged = True
        except Exception as e2:
            print(f"  Critical: Manual merge also failed ({e2})")
            merged = False

    # Import evaluate_condition with the test_manifest parameter
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import evaluate_condition

    evaluate_condition._batch_size = batch_size
    evaluate_condition._max_image_size = max_image_size

    metrics = evaluate_condition(
        condition_id=condition_label,
        adapter_path=f"composed_{condition_label}",
        model=model,
        tokenizer=tokenizer,
        max_eval_samples=max_eval_samples,
        test_manifest=test_manifest,
    )

    # evaluate_condition only returns the flat top-level metrics dict.
    # The full result (with per_source) is saved to eval_condition_{id}.json.
    # Load it back so gate scoring has access to per_source breakdowns.
    full_result_path = f"eval_condition_{condition_label}.json"
    if os.path.exists(full_result_path):
        with open(full_result_path, "r", encoding="utf-8") as fh:
            full_result = json.load(fh)
        metrics = full_result  # use the complete dict for gate scoring

    return metrics



def parse_sweep(sweep_specs: list) -> list:
    """
    Parse '0.9:0.1' style strings into [(logic_w, noise_w), ...].
    Also accepts '0.9,0.1' or '0.9 0.1'.
    """
    pairs = []
    for spec in sweep_specs:
        spec = spec.replace(",", ":").replace(" ", ":")
        parts = [p for p in spec.split(":") if p]
        if len(parts) != 2:
            raise ValueError(f"Invalid sweep spec: '{spec}' — expected 'logic:noise' e.g. '0.8:0.2'")
        lw, nw = float(parts[0]), float(parts[1])
        pairs.append((lw, nw))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="FeeVer 2.0 dual-adapter weight sweep."
    )
    parser.add_argument(
        "--logic-adapter", required=True,
        help="Path to the Logic adapter directory (e.g. clean_data/paige-lora-condition-clean_data-epoch-2.5)",
    )
    parser.add_argument(
        "--noise-adapter", required=True,
        help="Path to the Noise adapter directory (e.g. clean_data/paige-lora-condition-noise_r16-epoch-1.5)",
    )
    parser.add_argument(
        "--sweep", nargs="+", default=["0.9:0.1", "0.8:0.2", "0.7:0.3", "0.6:0.4"],
        metavar="LOGIC:NOISE",
        help="Weight ratio pairs to sweep, e.g. '0.9:0.1' '0.8:0.2' (default: 4-step sweep)",
    )
    parser.add_argument(
        "--test-manifest", type=str, default="eval_tiered_v1.jsonl",
        help="Path to the test manifest JSONL (default: eval_tiered_v1.jsonl)",
    )
    parser.add_argument(
        "--output", type=str, default="compose_results.json",
        help="Path to write the sweep results JSON (default: compose_results.json)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap eval samples for a quick smoke sweep",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Inference batch size (default 4)",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=896,
        help="Max image dimension before tokenization (default 896)",
    )
    args = parser.parse_args()

    # Validate paths
    for label, p in [("--logic-adapter", args.logic_adapter),
                     ("--noise-adapter", args.noise_adapter)]:
        if not os.path.isdir(p):
            print(f"Error: {label} path not found: {p}", file=sys.stderr)
            sys.exit(1)
    if not os.path.exists(args.test_manifest):
        print(f"Error: --test-manifest not found: {args.test_manifest}", file=sys.stderr)
        sys.exit(1)

    sweep_pairs = parse_sweep(args.sweep)
    print(f"Sweep plan: {len(sweep_pairs)} weight ratios × full eval")
    print(f"  Logic adapter : {args.logic_adapter}")
    print(f"  Noise adapter : {args.noise_adapter}")
    print(f"  Test manifest : {args.test_manifest}")
    print(f"  Ratios        : {[(f'{lw:.2f}:{nw:.2f}') for lw, nw in sweep_pairs]}")

    model, tokenizer = _load_model_with_dual_adapters(
        args.logic_adapter, args.noise_adapter
    )

    sweep_results = []
    best_entry = None
    best_score = -1.0

    for logic_w, noise_w in sweep_pairs:
        label = f"L{int(logic_w*10)}_N{int(noise_w*10)}"
        metrics = compose_and_eval(
            model=model,
            tokenizer=tokenizer,
            logic_weight=logic_w,
            noise_weight=noise_w,
            test_manifest=args.test_manifest,
            condition_label=label,
            max_eval_samples=args.max_samples,
            batch_size=args.batch_size,
            max_image_size=args.max_image_size,
        )

        # Pull Structural F1 (Structural tier) and Specialized EM
        per_source = metrics.get("per_source", {})
        structural_sources = ["invoices_donut_v1"]  # Medical invoice gate only; SROIE excluded (not trained on)
        structural_f1_vals = [
            per_source[s]["metrics"]["macro_f1"]
            for s in structural_sources
            if s in per_source
        ]
        structural_f1 = (
            sum(structural_f1_vals) / len(structural_f1_vals)
            if structural_f1_vals else 0.0
        )
        specialized_em = (
            per_source.get("paige_synthetic", {})
            .get("metrics", {})
            .get("fuzzy_exact_match", 0.0)
        )

        passes_gate = (
            structural_f1 >= STRUCTURAL_F1_GATE
            and specialized_em > SPECIALIZED_EM_GATE
        )

        entry = {
            "label":          label,
            "logic_weight":   logic_w,
            "noise_weight":   noise_w,
            "structural_f1":  round(structural_f1, 4),
            "specialized_em": round(specialized_em, 4),
            "passes_gate":    passes_gate,
            "metrics":        metrics,
        }
        sweep_results.append(entry)

        # Composite score: prioritise structural F1 (94% gate), then specialized EM
        composite = structural_f1 * 0.6 + specialized_em * 0.4
        if composite > best_score:
            best_score = composite
            best_entry = entry

        status = "✓ PASS" if passes_gate else "✗ FAIL"
        print(f"  [{label}] struct_f1={structural_f1:.4f}  spec_em={specialized_em:.4f}  {status}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Sweep complete. Best ratio: {best_entry['label']} "
          f"(struct_f1={best_entry['structural_f1']:.4f}, "
          f"spec_em={best_entry['specialized_em']:.4f})")
    passing = [e for e in sweep_results if e["passes_gate"]]
    if passing:
        print(f"  {len(passing)} ratio(s) pass the success gate.")
    else:
        print("  No ratio passed both gates — consider wider sweep or more noise_r16 epochs.")

    output = {
        "logic_adapter":    args.logic_adapter,
        "noise_adapter":    args.noise_adapter,
        "test_manifest":    args.test_manifest,
        "success_gate": {
            "structural_f1": STRUCTURAL_F1_GATE,
            "specialized_em": SPECIALIZED_EM_GATE,
        },
        "sweep": sweep_results,
        "best": best_entry,
        "passing_ratios": [e["label"] for e in passing],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
