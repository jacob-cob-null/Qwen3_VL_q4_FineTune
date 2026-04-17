import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

TARGET_FIELDS = [
    "date",
    "patient_name",
    "philhealth_number",
    "diagnosis_code",
    "procedure_code",
    "total_amount",
    "philhealth_benefit",
    "balance_due",
]

ALLOWED_DATASETS = {
    "cord_v2",
    "invoices_donut_v1",
    "paige_synthetic",
    "sroie_2019_v2",
}

SYSTEM_PROMPT = (
    "You are a medical billing document parser for Philippine hospital documents. "
    "Extract key-value pairs and return valid JSON only. No explanation, no markdown."
)

USER_PROMPT = (
    "Extract all billing fields. Return JSON with: "
    "date, patient_name, philhealth_number, diagnosis_code, procedure_code, "
    "total_amount, philhealth_benefit, balance_due. "
    "Use null for missing fields."
)


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    return str(value).strip()


def normalize_amount(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip().replace("P", "").replace("p", "").replace(",", "")
    return text if text else None


def normalize_ground_truth(raw: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in TARGET_FIELDS:
        val = raw.get(key)
        if key in {"total_amount", "philhealth_benefit", "balance_due"}:
            normalized[key] = normalize_amount(val)
        else:
            normalized[key] = normalize_scalar(val)
    return normalized


def validate_record(record: Dict[str, Any]) -> None:
    dataset = record.get("source_dataset")
    split = record.get("split")

    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Unsupported source_dataset: {dataset}")

    if dataset == "sroie_2019_v2" and split in {"train", "val"}:
        raise ValueError("SROIE is test-only and cannot be used in train/val splits")

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")

    gt = record.get("ground_truth")
    if not isinstance(gt, dict):
        raise ValueError("ground_truth must be an object")

    missing = [f for f in TARGET_FIELDS if f not in gt]
    if missing:
        raise ValueError(f"Missing required ground_truth fields: {missing}")



def build_messages(image_path: str, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": json.dumps(ground_truth, ensure_ascii=False),
        },
    ]



def canonicalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(raw)
    image_path = str(record.get("image_path", "")).strip()
    if not image_path:
        raise ValueError("image_path is required")

    ground_truth = normalize_ground_truth(record.get("ground_truth", {}))
    record["ground_truth"] = ground_truth
    record["messages"] = build_messages(image_path=image_path, ground_truth=ground_truth)

    validate_record(record)
    return record



def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_no}: {error}") from error



def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical golden JSONL for VLM fine-tuning")
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    canonical_records = []
    for raw in iter_jsonl(input_path):
        canonical_records.append(canonicalize_record(raw))

    write_jsonl(output_path, canonical_records)
    print(f"Wrote {len(canonical_records)} canonical records to {output_path}")



if __name__ == "__main__":
    main()
