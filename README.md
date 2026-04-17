# pAIge: Multimodal Fine-Tuning Pipeline

A minimal, high-performance pipeline for fine-tuning **Qwen/Qwen3-VL-4B-Instruct** on medical billing documents using **Unsloth**. Optimized for Windows environments with NVIDIA RTX 3060 GPUs.

## 🚀 Quick Start

1.  **Environment Setup**
    Ensure you have an NVIDIA GPU with 12GB+ VRAM (e.g., RTX 3060).
    ```powershell
    # Install dependencies and set up environment
    .\setup_env.ps1
    ```

2.  **GPU Verification**
    Validate your hardware and CUDA setup before starting long runs.
    ```bash
    python verify_gpu.py
    ```

3.  **Run Smoke Test**
    Verifies the entire data-loading, training, and evaluation pipeline on a small subset of data.
    ```bash
    python train.py --smoke-test
    ```

## 🛠 Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `train.py` | **Main Entry Point.** Handles ablation studies (Conditions A-G) varying dataset size and synthetic ratios. |
| `evaluate.py` | **Evaluation Engine.** Computes field-level Precision/Recall/F1 metrics on SROIE-formatted test sets. |
| `verify_gpu.py` | Hardware diagnostics and VRAM headroom analysis for RTX 3060. |
| `preflight.py` | Dataset integrity check; ensures images exist and JSONL paths are resolvable. |
| `report.py` | Aggregates metrics from multiple `eval_condition_*.json` files into a final summary. |
| `Datasets/` | Contains `Training_Data/golden` (merged sources) and `Testing_Data/sroie_2019_v2`. |
| `scripts/` | Internal utilities, including multimodal preparation patches. |

## 🧬 Training Configuration

The pipeline supports automated ablation runs across multiple conditions:

| Condition | Samples | Synthetic Ratio | Description |
| :--- | :--- | :--- | :--- |
| **A** | 500 | 30% | Small baseline |
| **B** | 500 | 40% | Higher synthetic ratio |
| **G** | 2000 | 0% | No synthetic data |

Run specific conditions using:
```bash
python train.py --id A B G
```

## 📊 Evaluation Schema

The model extracts 8 target fields from hospital Statements of Account (SOA):
- `date`, `patient_name`, `philhealth_number`, `diagnosis_code`, `procedure_code`, `total_amount`, `philhealth_benefit`, `balance_due`.

Metrics are saved to `eval_condition_{ID}.json` containing exact matches and macro-F1 scores.

## 🔧 Technical Notes

- **Optimizations**: Uses Unsloth for 2x faster training and 4-bit quantization to fit on consumer hardware.
- **Path Resolution**: Includes a robust `resolve_image_path` utility to handle absolute/relative path shifts across different machine environments.
- **Windows Support**: Includes global `TORCHDYNAMO_DISABLE` and float8 fallback patches to ensure stability on Windows.
