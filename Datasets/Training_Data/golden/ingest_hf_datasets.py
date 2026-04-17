import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset
from dotenv import load_dotenv


def configure_hf_token() -> None:
    # Ensure CLI runs can authenticate even when shell does not auto-load .env.
    repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(repo_root / ".env", override=True)

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)


def parse_ground_truth_blob(blob: Any) -> Dict[str, Any]:
    if blob is None:
        return {}
    if isinstance(blob, dict):
        return blob
    if isinstance(blob, str):
        text = blob.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def find_first_value(obj: Dict[str, Any], candidates: Iterable[str]) -> Optional[Any]:
    for key in candidates:
        if key in obj and obj[key] not in (None, ""):
            return obj[key]
    return None


def normalize_total(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().replace("P", "").replace("p", "").replace(",", "")
    return text or None


def extract_common_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    # SROIE style
    objects = sample.get("objects")
    if isinstance(objects, dict):
        entities = objects.get("entities")
        if isinstance(entities, dict):
            total = find_first_value(entities, ["total", "amount", "grand_total"])
            return {
                "date": find_first_value(entities, ["date", "invoice_date"]),
                "patient_name": None,
                "philhealth_number": None,
                "diagnosis_code": None,
                "procedure_code": None,
                "total_amount": normalize_total(total),
                "philhealth_benefit": None,
                "balance_due": normalize_total(total),
            }

    gt = parse_ground_truth_blob(sample.get("ground_truth"))
    gt_parse = gt.get("gt_parse") if isinstance(gt, dict) else None
    if isinstance(gt_parse, dict):
        gt = gt_parse

    # Handle CORD v2 nested total structure: gt_parse.total.total_price
    total_obj = gt.get("total")
    if isinstance(total_obj, dict):
        total = find_first_value(
            total_obj,
            ["total_price", "total_amount", "total", "grand_total"],
        )
    elif isinstance(total_obj, (str, int, float)):
        total = total_obj
    else:
        total = find_first_value(
            gt,
            [
                "total_amount",
                "amount_total",
                "grand_total",
                "total_price",
            ],
        )

    patient = find_first_value(
        gt,
        [
            "patient_name",
            "customer",
            "customer_name",
            "name",
            "recipient",
        ],
    )

    return {
        "date": find_first_value(gt, ["date", "invoice_date", "issued_date"]),
        "patient_name": str(patient).strip() if patient is not None else None,
        "philhealth_number": find_first_value(
            gt, ["philhealth_number", "philhealth_no", "member_id"]
        ),
        "diagnosis_code": find_first_value(gt, ["diagnosis_code", "icd_code"]),
        "procedure_code": find_first_value(gt, ["procedure_code", "cpt_code"]),
        "total_amount": normalize_total(total),
        "philhealth_benefit": normalize_total(
            find_first_value(gt, ["philhealth_benefit", "benefit_amount"])
        ),
        "balance_due": normalize_total(find_first_value(gt, ["balance_due", "amount_due", "due"])),
    }


def save_image(image_obj: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image_obj, "save"):
        image_obj.save(output_path)
        return
    raise ValueError(f"Unsupported image object type: {type(image_obj)}")


def record_template(
    source_dataset: str,
    split: str,
    sample_id: str,
    image_path: str,
    hf_source_id: str,
    hf_split: str,
    hf_index: int,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_dataset": source_dataset,
        "split": split,
        "is_synthetic": False,
        "seed": 42,
        "image_path": image_path.replace("\\", "/"),
        "metadata": {
            "template_id": None,
            "generator": "huggingface",
            "source_reference_id": f"{hf_source_id}:{hf_split}:{hf_index}",
            "notes": "Auto-ingested from Hugging Face dataset",
        },
        "ground_truth": ground_truth,
    }


def ingest_one_dataset(
    hf_id: str,
    hf_split: str,
    source_dataset: str,
    target_split: str,
    base_dir: Path,
    max_samples: Optional[int],
) -> int:
    use_streaming = max_samples is not None
    hf_token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(
        hf_id,
        split=hf_split,
        streaming=use_streaming,
        token=hf_token,
    )

    raw_dir = base_dir / "sources" / source_dataset / "raw"
    img_dir = base_dir / "sources" / source_dataset / "images" / target_split
    raw_path = raw_dir / f"{target_split}.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with raw_path.open("w", encoding="utf-8") as outfile:
        for idx, sample in enumerate(dataset):
            if max_samples is not None and idx >= max_samples:
                break

            image_obj = sample.get("image")
            if image_obj is None:
                continue

            sample_id = f"{source_dataset}_{target_split}_{idx:06d}"
            image_file = img_dir / f"{sample_id}.png"
            save_image(image_obj, image_file)

            ground_truth = extract_common_fields(sample)
            record = record_template(
                source_dataset=source_dataset,
                split=target_split,
                sample_id=sample_id,
                image_path=str(image_file),
                hf_source_id=hf_id,
                hf_split=hf_split,
                hf_index=idx,
                ground_truth=ground_truth,
            )
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    configure_hf_token()

    parser = argparse.ArgumentParser(description="Ingest selected Hugging Face datasets into golden raw JSONL")
    parser.add_argument(
        "--base-dir",
        default="Datasets/Training_Data/golden",
        help="Golden dataset base directory",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max samples per dataset for smoke testing",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    plan = [
        {
            "hf_id": "naver-clova-ix/cord-v2",
            "hf_split": "train",
            "source_dataset": "cord_v2",
            "target_split": "train",
        },
        {
            "hf_id": "katanaml-org/invoices-donut-data-v1",
            "hf_split": "train",
            "source_dataset": "invoices_donut_v1",
            "target_split": "train",
        },
        {
            "hf_id": "rth/sroie-2019-v2",
            "hf_split": "test",
            "source_dataset": "sroie_2019_v2",
            "target_split": "test",
        },
    ]

    total = 0
    for item in plan:
        count = ingest_one_dataset(
            hf_id=item["hf_id"],
            hf_split=item["hf_split"],
            source_dataset=item["source_dataset"],
            target_split=item["target_split"],
            base_dir=base_dir,
            max_samples=args.max_samples,
        )
        total += count
        print(f"{item['source_dataset']}: wrote {count} records")

    print(f"Total records written: {total}")


if __name__ == "__main__":
    main()
