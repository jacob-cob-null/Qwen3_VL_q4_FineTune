import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

ALLOWED_SPLITS = {"train", "val", "test"}


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_no}: {error}") from error


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_row(row: Dict) -> None:
    dataset = row.get("source_dataset")
    split = row.get("split")

    if not dataset:
        raise ValueError("source_dataset is required")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {sorted(ALLOWED_SPLITS)}, got: {split}")


def organize(input_paths: List[Path], base_dir: Path, exclude_datasets: Set[str]) -> None:
    by_dataset_split: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    by_split: Dict[str, List[Dict]] = defaultdict(list)

    total = 0
    for input_path in input_paths:
        for row in iter_jsonl(input_path):
            validate_row(row)
            dataset = row["source_dataset"]
            if dataset in exclude_datasets:
                continue
            split = row["split"]
            by_dataset_split[dataset][split].append(row)
            by_split[split].append(row)
            total += 1

    for dataset, split_map in by_dataset_split.items():
        for split, rows in split_map.items():
            out = base_dir / "sources" / dataset / "canonical" / f"{split}.jsonl"
            write_jsonl(out, rows)

    merged_dir = base_dir / "merged"
    for split, rows in by_split.items():
        out = merged_dir / f"all_sources_{split}.jsonl"
        write_jsonl(out, rows)

    # One combined train+val pool for ablation condition sampling.
    train_rows = by_split.get("train", [])
    val_rows = by_split.get("val", [])
    write_jsonl(merged_dir / "all_sources_train_val.jsonl", train_rows + val_rows)

    print(f"Organized {total} records from {len(input_paths)} file(s) into sources/*/canonical and merged/ files under {base_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize canonical golden JSONL into dataset/split folders")
    parser.add_argument("--input", help="Single canonical JSONL path")
    parser.add_argument(
        "--input-glob",
        help="Glob pattern for canonical JSONL files (example: Datasets/Training_Data/golden/sources/*/canonical/*.jsonl)",
    )
    parser.add_argument(
        "--base-dir",
        default="Datasets/Training_Data/golden",
        help="Golden base directory where sources/ and merged/ are located",
    )
    parser.add_argument(
        "--exclude-datasets",
        default="",
        help="Comma-separated dataset names to exclude from output",
    )
    args = parser.parse_args()

    if not args.input and not args.input_glob:
        raise ValueError("Provide either --input or --input-glob")

    input_paths: List[Path] = []

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_paths.append(input_path)

    if args.input_glob:
        matched = sorted(Path().glob(args.input_glob))
        if not matched:
            raise FileNotFoundError(f"No files matched input glob: {args.input_glob}")
        input_paths.extend(matched)

    unique_paths = sorted(set(input_paths))
    exclude_datasets = {
        item.strip()
        for item in str(args.exclude_datasets).split(",")
        if item.strip()
    }

    base_dir = Path(args.base_dir)
    organize(input_paths=unique_paths, base_dir=base_dir, exclude_datasets=exclude_datasets)


if __name__ == "__main__":
    main()
