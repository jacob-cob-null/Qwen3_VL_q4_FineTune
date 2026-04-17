# Golden JSON Training Template (pAIge FT Reference)

This folder defines a canonical JSON format for fine-tuning data samples.

## Scope

- Model: Qwen/Qwen3-VL-4B-Instruct
- Trainer: Unsloth + TRL SFTTrainer
- Task: Philippine medical billing KVP extraction
- Rule: SROIE is test-only and must never be in any training split

## Required Output Fields

The assistant extraction target is:

- date
- patient_name
- philhealth_number
- diagnosis_code
- procedure_code
- total_amount
- philhealth_benefit
- balance_due

Use null for missing values.

## Golden Sample Structure

Each sample is one JSON object with:

- sample_id: stable unique id
- source_dataset: cord_v2 | invoices_donut_v1 | paige_synthetic | sroie_2019_v2
- split: train | val | test
- is_synthetic: boolean
- seed: integer
- image_path: path to image
- metadata: source-specific metadata and reproducibility tags
- ground_truth: canonical target object
- messages: chat-format sample ready for VLM SFT

## Hard Rules

- Reject any sample where source_dataset == sroie_2019_v2 and split in {train, val}.
- Keep vision encoder frozen across all ablation conditions (handled in training config, not data).
- Generate all 800 synthetic documents before any training run starts.
- Keep seeds fixed for reproducibility.

## Files

- golden_sample_template.json: canonical empty template
- golden_examples.jsonl: valid example rows
- build_golden_jsonl.py: converter/validator scaffold to emit training-ready JSONL
- organize_golden_jsonl.py: organizer that writes per-dataset/per-split outputs

## Folder Layout

This folder is organized as follows:

- sources/cord_v2/raw
- sources/cord_v2/canonical
- sources/invoices_donut_v1/raw
- sources/invoices_donut_v1/canonical
- sources/paige_synthetic/raw
- sources/paige_synthetic/canonical
- sources/sroie_2019_v2/raw
- sources/sroie_2019_v2/canonical
- merged
- conditions

Use raw/ for dataset-specific pre-canonical JSONL inputs.
Use canonical/ for cleaned records emitted by build_golden_jsonl.py.
Use merged/ for combined train/val/test files used by training and evaluation jobs.

## Recommended Commands

1. Canonicalize each dataset input:

python Datasets/Training_Data/golden/build_golden_jsonl.py --input Datasets/Training_Data/golden/sources/paige_synthetic/raw/train.jsonl --output Datasets/Training_Data/golden/sources/paige_synthetic/canonical/train.jsonl

2. Organize canonical files from all dataset folders (no manual merge needed):

python Datasets/Training_Data/golden/organize_golden_jsonl.py --input-glob "Datasets/Training_Data/golden/sources/_/canonical/_.jsonl" --base-dir Datasets/Training_Data/golden

3. Generate Azure-ready dataset manifest (counts, missing images, JSONL hashes):

python Datasets/Training*Data/golden/generate_dataset_manifest.py --jsonl-glob "Datasets/Training_Data/golden/merged/all_sources*\*.jsonl" --dataset-root . --output Datasets/Training_Data/golden/manifest.dataset.json
