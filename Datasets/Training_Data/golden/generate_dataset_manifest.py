import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_no}")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        while True:
            chunk = infile.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(dataset_root: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return candidate
    return dataset_root / candidate


def build_manifest(jsonl_paths: List[Path], dataset_root: Path) -> Dict[str, Any]:
    files_section: List[Dict[str, Any]] = []
    totals = {
        "records": 0,
        "missing_images": 0,
        "files": len(jsonl_paths),
    }

    by_dataset = defaultdict(int)
    by_split = defaultdict(int)

    for path in jsonl_paths:
        file_records = 0
        file_missing_images = 0
        missing_examples: List[str] = []

        for row in iter_jsonl(path):
            file_records += 1
            totals["records"] += 1

            source_dataset = str(row.get("source_dataset", "unknown"))
            split = str(row.get("split", "unknown"))
            by_dataset[source_dataset] += 1
            by_split[split] += 1

            image_path = str(row.get("image_path", "")).strip()
            if not image_path:
                file_missing_images += 1
                totals["missing_images"] += 1
                if len(missing_examples) < 50:
                    missing_examples.append("<empty image_path>")
                continue

            resolved = resolve_image_path(dataset_root=dataset_root, image_path=image_path)
            if not resolved.exists():
                file_missing_images += 1
                totals["missing_images"] += 1
                if len(missing_examples) < 50:
                    missing_examples.append(image_path)

        files_section.append(
            {
                "path": str(path).replace("\\", "/"),
                "sha256": sha256_file(path),
                "records": file_records,
                "missing_images": file_missing_images,
                "missing_image_examples": missing_examples,
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root).replace("\\", "/"),
        "totals": totals,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_split": dict(sorted(by_split.items())),
        "files": files_section,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset manifest for Azure fine-tuning readiness")
    parser.add_argument(
        "--jsonl-glob",
        default="Datasets/Training_Data/golden/merged/all_sources_*.jsonl",
        help="Glob for JSONL files to include",
    )
    parser.add_argument(
        "--dataset-root",
        default=".",
        help="Root path for resolving relative image paths",
    )
    parser.add_argument(
        "--output",
        default="Datasets/Training_Data/golden/manifest.dataset.json",
        help="Manifest output JSON path",
    )
    parser.add_argument(
        "--strict-missing-images",
        action="store_true",
        help="Exit with error if any image_path is missing",
    )
    args = parser.parse_args()

    jsonl_paths = sorted(Path().glob(args.jsonl_glob))
    if not jsonl_paths:
        raise FileNotFoundError(f"No files matched --jsonl-glob: {args.jsonl_glob}")

    dataset_root = Path(args.dataset_root).resolve()
    manifest = build_manifest(jsonl_paths=jsonl_paths, dataset_root=dataset_root)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(manifest, outfile, indent=2, ensure_ascii=False)
        outfile.write("\n")

    missing = int(manifest["totals"]["missing_images"])
    print(f"Wrote manifest to {output_path} with {manifest['totals']['records']} records and {missing} missing images")

    if args.strict_missing_images and missing > 0:
        raise SystemExit("Missing images detected while --strict-missing-images is enabled")


if __name__ == "__main__":
    main()
