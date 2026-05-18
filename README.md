# pAIge: Multimodal Fine-Tuning Pipeline

Fine-tuning **Qwen/Qwen3-VL-4B-Instruct** on Philippine hospital Statements of Account (SOA) for structured field extraction. Built with Unsloth for 4-bit quantized training on a single RTX 3060 (12 GB VRAM).

## Requirements

- Windows 10/11
- Python 3.11
- NVIDIA GPU with 12 GB+ VRAM (tested on RTX 3060)
- CUDA 12.1

## Setup

**1. Run the setup script**

```powershell
powershell -ExecutionPolicy Bypass -File setup_env.ps1
```

This creates `.venv311`, installs PyTorch 2.5.1+cu121, then installs everything in `requirements.txt`.

Unsloth is not in `requirements.txt` because it requires PyTorch to be present first. The script handles this ordering automatically. If you need to reinstall Unsloth manually:

```powershell
.venv311\Scripts\python.exe -m pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
```

**2. Activate the environment**

```powershell
.\.venv311\Scripts\Activate.ps1
```

**3. Verify GPU**

```bash
python verify_gpu.py
```

Expects CUDA 12.1 available and ~12 GB VRAM free.

## Workflow

**Train**

```bash
python train.py --id clean_data
```

Adapters are saved to `./paige-lora-condition-{id}/`. Use `--smoke-test` to validate the pipeline on 10 steps before committing to a full run.

**Evaluate**

```bash
python evaluate.py --adapter clean_data/paige-lora-condition-clean_data-epoch-2.5 \
                   --condition-id my_run
```

Outputs `eval_condition_my_run.json` with per-field P/R/F1 and a tier summary.

**Compose dual adapters**

The optimal adapters are stored in `optimal/`. See `optimal/setup.md` for full specifications.

```bash
python compose_adapters.py \
    --logic-adapter optimal/logic \
    --noise-adapter optimal/noise \
    --condition-label L9_N0
```

Sweeps weight ratios and saves the best composed adapter.

**Aggregate results**

```bash
python scripts/compile_results.py
```

Writes `master_results.csv` summarising all `eval_condition_*.json` files.

## Repository Structure

| Path | Description |
| :--- | :--- |
| `train.py` | Training entry point — ablation conditions, LoRA config, epoch callbacks |
| `evaluate.py` | Inference + per-field metrics (P/R/F1, fuzzy EM, tier summary) |
| `compose_adapters.py` | Merges Logic + Noise adapters via PEFT `add_weighted_adapter` |
| `make_eval_tiered.py` | Builds stratified test manifests (synthetic / invoice / SROIE tiers) |
| `preflight.py` | Pre-run checks: image paths, data integrity, VRAM headroom |
| `verify_gpu.py` | CUDA diagnostics |
| `report.py` | Plots training loss and validation F1 curves |
| `scripts/` | Prompts, post-processing, result aggregation, rescoring utilities |
| `Datasets/` | `Training_Data/golden/merged/` (JSONL) + `Testing_Data/sroie_2019_v2/` |
| `optimal/` | Best Logic + Noise adapters (97:3 ratio); see `optimal/setup.md` |
| `clean_data/` | Logic adapter checkpoints (epoch 1.0 – 3.0, excluding 1.5 moved to optimal/) |
| `clean_result_raw/` | Eval outputs for single-adapter epoch sweep |
| `clean_result_dual/` | Eval outputs for composed dual-adapter runs |

## Extraction Schema

The model extracts 8 fields from each document:

| Tier | Fields |
| :--- | :--- |
| Temporal | `date` |
| Financial | `total_amount`, `balance_due`, `philhealth_benefit` |
| Clinical | `patient_name`, `philhealth_number`, `diagnosis_code`, `procedure_code` |

Outputs are JSON strings validated against this schema. Post-processing (`scripts/postprocess.py`) applies deterministic fixes before scoring: schema enforcement, `balance_due` imputation for non-medical sources, and date anchor error flagging.

## Best Result

`eval_condition_L9_N0.json` — composed adapter at **97% Logic / 3% Noise** weight:

| Metric | Value |
| :--- | :--- |
| Fuzzy exact match | 0.3894 |
| Macro F1 | 0.6206 |
| Structural F1 | 0.9335 |

| Field | F1 |
| :--- | :--- |
| `total_amount` | 0.916 |
| `date` | 0.798 |
| `patient_name` | 0.608 |
| `balance_due` | 0.124 |

Evaluated on 226 SROIE test samples. Clinical/PhilHealth fields score 0.0 on this set (no ground-truth coverage — SROIE is non-medical receipts). See `optimal/setup.md` for full adapter specifications.

## Technical Notes

- **4-bit quantization** via Unsloth `FastVisionModel` keeps peak VRAM under 11 GB during training
- **Dual-adapter strategy**: Logic adapter (clean documents, LoRA r=32) + Noise adapter (degraded scans, r=32, vision encoder unfrozen), merged at inference via PEFT
- **Windows stability**: `TORCHDYNAMO_DISABLE=1` set globally; float8 dtype falls back to bfloat16
- **Instruction alignment**: unified extraction prompt in `scripts/prompts.py` shared between `train.py` and `evaluate.py` to prevent train/eval drift
