"""Create verified sequence-aware FOD-A train, validation and test manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from fod_detection.data.leakage_resistant_split import (
    assign_groups,
    audit_official_partitions,
    build_sequence_groups,
    load_records,
    summarise_conditions,
    summarise_sequences,
    verify_split,
    write_outputs,
)


DEFAULT_VOC_ROOT = Path(
    "FOD-A Dataset/FODPascalVOCFormat-V.2.1/VOC2007"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sequence-aware, leakage-resistant FOD-A split."
    )
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits/final"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-dhash-distance", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }

    records = load_records(args.voc_root)
    groups = build_sequence_groups(records, args.max_dhash_distance)
    assignments = assign_groups(groups, ratios, args.seed)
    verification = verify_split(records, groups, assignments)

    summary = {
        "method": (
            "Consecutive image IDs remain together when weather, lighting, object "
            "labels and perceptual similarity agree. Whole groups are assigned "
            "within each weather-light stratum."
        ),
        "parameters": {
            "ratios": ratios,
            "seed": args.seed,
            "max_dhash_distance": args.max_dhash_distance,
        },
        "verification": verification,
        "official_partition_repairs": audit_official_partitions(
            args.voc_root,
            records,
        ),
        "sequence_statistics": summarise_sequences(
            records,
            groups,
            args.max_dhash_distance,
        ),
        "condition_counts": summarise_conditions(groups, assignments),
    }
    write_outputs(args.output_dir, groups, assignments, summary)

    print(f"Created {verification['sequence_count']:,} sequence groups")
    print(f"Partition counts: {verification['partition_counts']}")
    print(f"Wrote verified manifests to {args.output_dir}")


if __name__ == "__main__":
    main()
