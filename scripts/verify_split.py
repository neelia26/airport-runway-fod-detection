"""Independently check a generated split manifest for leakage and omissions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a FOD-A split manifest.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("data/splits/final/manifest.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "FOD-A Dataset/FODPascalVOCFormat-V.2.1/VOC2007/Annotations"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    manifest_ids = [row["image_id"] for row in rows]
    expected_ids = {path.stem for path in args.annotations.glob("*.xml")}
    sequence_partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sequence_partitions[row["sequence_id"]].add(row["split"])

    checks = {
        "all_annotation_ids_present": set(manifest_ids) == expected_ids,
        "no_duplicate_ids": len(manifest_ids) == len(set(manifest_ids)),
        "valid_partition_names": {row["split"] for row in rows}
        == {"train", "validation", "test"},
        "no_sequence_crosses_partitions": all(
            len(partitions) == 1 for partitions in sequence_partitions.values()
        ),
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    if not all(checks.values()):
        raise SystemExit("Split verification failed")

    print(f"PASS: {len(rows):,} images in {len(sequence_partitions):,} isolated groups")


if __name__ == "__main__":
    main()
