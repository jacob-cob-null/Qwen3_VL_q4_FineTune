# Third-Party Notices and Data Provenance

## Scope

`LICENSE` applies only to original source code in this repository. It does not
relicense model weights, adapter weights, external datasets, document images,
or generated outputs.

## Base model and training stack

The adapters in `optimal/` are LoRA derivatives of
`unsloth/qwen3-vl-4b-instruct-unsloth-bnb-4bit`, based on Qwen3-VL. Users must
review and comply with the applicable upstream model licenses and use terms:

- Qwen: <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>
- Unsloth model artifact: <https://huggingface.co/unsloth/qwen3-vl-4b-instruct-unsloth-bnb-4bit>
- Unsloth library: <https://github.com/unslothai/unsloth>

The adapter files are not independently relicensed by this repository. The
adapter model cards intentionally refer back to this notice and the upstream
terms.

## External datasets

The dataset manifests and JSONL examples reference these upstream datasets:

- CORD v2: <https://huggingface.co/datasets/naver-clova-ix/cord-v2>
- Invoices-Donut v1: <https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1>
- SROIE 2019 v2: <https://huggingface.co/datasets/rth/sroie-2019-v2>

These datasets retain their own licenses, attribution requirements, and
redistribution restrictions. This repository does not grant permission to
redistribute their images or annotations. Verify the current upstream terms
before commercial use, publication, or redistribution.

## Synthetic data

`paige_synthetic` records are project-generated synthetic examples. The
pipeline uses Faker, ReportLab, Augraphy, and Pillow; those dependencies have
separate licenses documented by their respective projects. Synthetic records
should not be represented as real hospital records or as evidence of clinical
performance.

## Sensitive-data boundary

The extraction schema contains medical-style fields and identifiers. Before
sharing any derived dataset, predictions, screenshots, or logs, confirm that
no real patient information is present and apply the relevant privacy and
security review. Copyright clearance does not establish privacy clearance.
