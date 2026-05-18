# Optimal Adapter Configuration

This directory contains the two LoRA adapters that produced the best evaluation result
when composed at a **97:3 (Logic:Noise)** weight ratio.

## Adapter Summary

| | Logic | Noise |
| :--- | :--- | :--- |
| **Path** | `optimal/logic/` | `optimal/noise/` |
| **Condition** | `clean_data` | `noise_r32` |
| **Epoch** | 1.5 | 8 |
| **LoRA rank (r)** | 32 | 32 |
| **LoRA alpha** | 64 | 64 |
| **Dropout** | 0 | 0 |
| **Target layers** | Language/text only | Language + vision encoder |
| **Training samples** | 1,200 (67% synthetic) | 800 (100% high-noise synthetic) |
| **Base model** | `unsloth/qwen3-vl-4b-instruct-unsloth-bnb-4bit` | same |

**Key difference**: the Noise adapter unfreezes the vision encoder (`vision|image|visual|patch` added to `target_modules`), making it sensitive to degraded/noisy document scans. The Logic adapter targets language layers only for clean structured extraction.

## Composition

Merged via PEFT `add_weighted_adapter`:

```
composed = 0.97 × Logic + 0.03 × Noise
```

**Evaluation results** on 226 SROIE test samples after composition:

| Metric | Value |
| :--- | :--- |
| Fuzzy exact match | 0.3894 |
| Macro F1 | 0.6206 |
| Structural F1 | 0.9335 |
| Specialized EM | 0.38 |

Per-field breakdown (Logic+Noise composed):

| Field | F1 |
| :--- | :--- |
| `total_amount` | 0.916 |
| `date` | 0.798 |
| `patient_name` | 0.608 |
| `balance_due` | 0.124 |
| `philhealth_*` / clinical | 0.0 (no GT in SROIE) |

> Note: `passes_gate` is `False` — the structural F1 gate (0.94) was not met (0.9335 achieved).
> The 97:3 ratio was the single best-performing configuration tested.

## Reproducing the Composition

```bash
python compose_adapters.py \
    --logic-adapter optimal/logic \
    --noise-adapter optimal/noise \
    --condition-label L9_N0
```

## Reproducing Evaluation

```bash
python evaluate.py \
    --adapter-path composed_L9_N0 \
    --condition-id L9_N0
```

## Retraining

If retraining from scratch, these adapters were produced by:

```bash
# Logic adapter
python train.py --id clean_data

# Noise adapter
python train.py --id noise_r32
```

The Logic adapter checkpoint at epoch 1.5 was selected (saved automatically by `SaveEpochsCallback`).
The Noise adapter ran for 8 epochs with the vision encoder unfrozen.
